package jpamb;

import java.lang.classfile.ClassFile;
import java.lang.classfile.ClassTransform;
import java.lang.classfile.CodeBuilder;
import java.lang.classfile.CodeElement;
import java.lang.classfile.CodeTransform;
import java.lang.classfile.MethodModel;
import java.lang.classfile.instruction.ArrayLoadInstruction;
import java.lang.classfile.instruction.BranchInstruction;
import java.lang.classfile.instruction.ConstantInstruction;
import java.lang.classfile.instruction.IncrementInstruction;
import java.lang.classfile.instruction.LoadInstruction;
import java.lang.classfile.instruction.OperatorInstruction;
import java.lang.classfile.instruction.StoreInstruction;
import java.lang.constant.ClassDesc;
import java.lang.constant.MethodTypeDesc;
import java.lang.instrument.ClassFileTransformer;
import java.lang.instrument.Instrumentation;
import java.security.ProtectionDomain;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.lang.classfile.TypeKind;

public class InstrumentationAgent {
  public static final Set<String> INSTRUMENTED_CLASSES = ConcurrentHashMap.newKeySet();
  public static final Set<String> OBSERVED_OUTCOMES = ConcurrentHashMap.newKeySet();
  public static final Set<String> COVERED_SITES = ConcurrentHashMap.newKeySet();
  public static final ConcurrentHashMap<String, IntValueStats> SITE_INT_STATS =
      new ConcurrentHashMap<>();

  private static final ClassDesc AGENT_DESC = ClassDesc.of("jpamb.InstrumentationAgent");
  private static final ClassDesc INT_DESC = ClassDesc.ofDescriptor("I");
  private static final ClassDesc STRING_DESC = ClassDesc.ofDescriptor("Ljava/lang/String;");
  private static final ClassDesc VOID_DESC = ClassDesc.ofDescriptor("V");
  private static final MethodTypeDesc RECORD_SITE_DESC =
      MethodTypeDesc.of(VOID_DESC, STRING_DESC);
  private static final MethodTypeDesc RECORD_INT_DESC =
      MethodTypeDesc.of(VOID_DESC, STRING_DESC, INT_DESC);

  public static final class IntValueStats {
    public final AtomicInteger count = new AtomicInteger(0);
    public volatile int min = Integer.MAX_VALUE;
    public volatile int max = Integer.MIN_VALUE;

    public synchronized void observe(int value) {
      int n = count.incrementAndGet();
      if (n == 1) {
        min = value;
        max = value;
      } else {
        if (value < min) {
          min = value;
        }
        if (value > max) {
          max = value;
        }
      }
    }
  }

public static void premain(String agentArgs, Instrumentation inst) {
    inst.addTransformer(
        new ClassFileTransformer() {
          @Override
          public byte[] transform(
              ClassLoader loader,
              String className,
              Class<?> classBeingRedefined,
              ProtectionDomain protectionDomain,
              byte[] classfileBuffer) {
            if (className == null || !className.startsWith("jpamb/cases/")) {
              return null;
            }
            try {
              String fqcn = className.replace('/', '.');
              INSTRUMENTED_CLASSES.add(fqcn);
              COVERED_SITES.add("class:" + fqcn);

              ClassFile cf = ClassFile.of();
              ClassTransform classTransform =
                  (classBuilder, classElement) -> {
                    if (classElement instanceof MethodModel methodModel) {
                      String methodName = methodModel.methodName().stringValue();
                      if ("<clinit>".equals(methodName)) {
                        classBuilder.with(classElement);
                        return;
                      }
                      String methodPrefix =
                          fqcn
                              + "."
                              + methodName
                              + methodModel.methodTypeSymbol().descriptorString();

                      classBuilder.transformMethod(
                          methodModel,
                          (mb, me) -> {
                            if (me instanceof java.lang.classfile.CodeModel codeModel) {
                              AtomicInteger siteSeq = new AtomicInteger(0);
                              CodeTransform codeTransform =
                                  (CodeBuilder cb, CodeElement ce) -> {
                                    if (ce instanceof BranchInstruction) {
                                      emitSite(cb, methodPrefix + "#jmp@" + siteSeq.getAndIncrement());
                                      cb.with(ce);
                                    } else if (ce instanceof LoadInstruction load
                                        && load.typeKind() == TypeKind.INT) {
                                      cb.with(ce);
                                      emitDupInt(
                                          cb,
                                          methodPrefix
                                              + "#iload"
                                              + load.slot()
                                              + "@"
                                              + siteSeq.getAndIncrement());
                                    } else if (ce instanceof StoreInstruction store
                                        && store.typeKind() == TypeKind.INT) {
                                      cb.with(ce);
                                      emitLocalInt(
                                          cb,
                                          store.slot(),
                                          methodPrefix
                                              + "#istore"
                                              + store.slot()
                                              + "@"
                                              + siteSeq.getAndIncrement());
                                    } else if (ce instanceof IncrementInstruction inc) {
                                      cb.with(ce);
                                      emitLocalInt(
                                          cb,
                                          inc.slot(),
                                          methodPrefix
                                              + "#iinc"
                                              + inc.slot()
                                              + "@"
                                              + siteSeq.getAndIncrement());
                                    } else if (ce instanceof ConstantInstruction c) {
                                      cb.with(ce);
                                      if (c.constantValue() instanceof Integer) {
                                        emitDupInt(
                                            cb,
                                            methodPrefix + "#const@" + siteSeq.getAndIncrement());
                                      }
                                    } else if (ce instanceof ArrayLoadInstruction arrLoad
                                        && arrLoad.typeKind() == TypeKind.INT) {
                                      emitDupInt(
                                          cb,
                                          methodPrefix
                                              + "#iaload_idx@"
                                              + siteSeq.getAndIncrement());
                                      cb.with(ce);
                                      emitDupInt(
                                          cb,
                                          methodPrefix + "#iaload_res@" + siteSeq.getAndIncrement());
                                    } else if (ce instanceof OperatorInstruction op
                                        && op.typeKind() == TypeKind.INT) {
                                      switch (op.opcode()) {
                                        case IDIV, IREM -> emitDupInt(
                                            cb,
                                            methodPrefix + "#div_rhs@" + siteSeq.getAndIncrement());
                                        default -> {}
                                      }
                                      cb.with(ce);
                                      switch (op.opcode()) {
                                        case IADD, ISUB, IMUL, IDIV, IREM -> emitDupInt(
                                            cb, methodPrefix + "#ires@" + siteSeq.getAndIncrement());
                                        default -> {}
                                      }
                                    } else {
                                      cb.with(ce);
                                    }
                                  };
                              mb.transformCode(codeModel, codeTransform);
                            } else {
                              mb.with(me);
                            }
                          });
                    } else {
                      classBuilder.with(classElement);
                    }
                  };
              return cf.transformClass(cf.parse(classfileBuffer), classTransform);
            } catch (Throwable t) {
              if (Boolean.getBoolean("jpamb.agent.debug")) {
                t.printStackTrace(System.err);
              }
              return null;
            }
          }
        });
  }

  private static void emitSite(CodeBuilder cb, String site) {
    cb.ldc(site);
    cb.invokestatic(AGENT_DESC, "recordSite", RECORD_SITE_DESC, false);
  }

  private static void emitDupInt(CodeBuilder cb, String site) {
    cb.dup();
    cb.ldc(site);
    cb.swap();
    cb.invokestatic(AGENT_DESC, "recordIntValue", RECORD_INT_DESC, false);
  }

  private static void emitLocalInt(CodeBuilder cb, int slot, String site) {
    cb.ldc(site);
    cb.iload(slot);
    cb.invokestatic(AGENT_DESC, "recordIntValue", RECORD_INT_DESC, false);
  }

  public static void recordSite(String site) {
    if (site != null && !site.isBlank()) {
      COVERED_SITES.add(site);
    }
  }

  public static void recordIntValue(String site, int value) {
    if (site == null || site.isBlank()) {
      return;
    }
    COVERED_SITES.add(site);
    SITE_INT_STATS.computeIfAbsent(site, k -> new IntValueStats()).observe(value);
  }

  public static void recordOutcome(String outcome) {
    if (outcome != null && !outcome.isBlank()) {
      OBSERVED_OUTCOMES.add(outcome);
    }
  }

  public static void reset() {
    INSTRUMENTED_CLASSES.clear();
    OBSERVED_OUTCOMES.clear();
    COVERED_SITES.clear();
    SITE_INT_STATS.clear();
  }

  public static String dumpSummary() {
    StringBuilder sb = new StringBuilder();
    sb.append("AGENT_SITES:").append(String.join("|", COVERED_SITES)).append('\n');
    sb.append("AGENT_INT_RANGES:");
    boolean first = true;
    for (Map.Entry<String, IntValueStats> entry : SITE_INT_STATS.entrySet()) {
      if (!first) {
        sb.append(';');
      }
      first = false;
      IntValueStats stats = entry.getValue();
      sb.append(entry.getKey())
          .append('=')
          .append(stats.min)
          .append("..")
          .append(stats.max)
          .append('#')
          .append(stats.count.get());
    }
    return sb.toString();
  }
}