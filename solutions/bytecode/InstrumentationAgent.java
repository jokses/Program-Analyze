package solutions.bytecode;

import java.io.BufferedWriter;
import java.io.IOException;
import java.lang.classfile.ClassBuilder;
import java.lang.classfile.ClassElement;
import java.lang.classfile.ClassFile;
import java.lang.classfile.ClassModel;
import java.lang.classfile.ClassTransform;
import java.lang.classfile.CodeBuilder;
import java.lang.classfile.CodeElement;
import java.lang.classfile.CodeTransform;
import java.lang.classfile.Instruction;
import java.lang.classfile.MethodModel;
import java.lang.classfile.MethodTransform;
import java.lang.classfile.Opcode;
import java.lang.classfile.TypeKind;
import java.lang.classfile.instruction.ArrayLoadInstruction;
import java.lang.classfile.instruction.BranchInstruction;
import java.lang.classfile.instruction.OperatorInstruction;
import java.lang.constant.ClassDesc;
import java.lang.constant.MethodTypeDesc;
import java.lang.instrument.ClassFileTransformer;
import java.lang.instrument.Instrumentation;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.ProtectionDomain;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;

public final class InstrumentationAgent {
    private static final ClassDesc AGENT_DESC =
            ClassDesc.of("solutions.bytecode.InstrumentationAgent");
    private static final MethodTypeDesc RECORD_SITE_DESC =
            MethodTypeDesc.ofDescriptor("(Ljava/lang/String;I)V");
    private static final MethodTypeDesc RECORD_VAL_DESC =
            MethodTypeDesc.ofDescriptor("(Ljava/lang/String;III)V");

    private static final Set<String> VISITED_SITES = ConcurrentHashMap.newKeySet();
    private static final ConcurrentHashMap<String, CopyOnWriteArrayList<Integer>> OBSERVED_VALUES =
            new ConcurrentHashMap<>();
    public static final Set<String> OBSERVED_OUTCOMES = ConcurrentHashMap.newKeySet();

    private InstrumentationAgent() {}

    public static void premain(String agentArgs, Instrumentation inst) {
        if (agentArgs != null && !agentArgs.isBlank()) {
            Path target = Path.of(agentArgs.trim());
            java.lang.Runtime.getRuntime().addShutdownHook(
                    new Thread(() -> writeOutput(target), "jpamb-agent-shutdown"));
        }

        inst.addTransformer(new ClassFileTransformer() {
            @Override
            public byte[] transform(
                    ClassLoader loader,
                    String className,
                    Class<?> classBeingRedefined,
                    ProtectionDomain domain,
                    byte[] classfileBuffer) {
                if (className == null || !className.startsWith("jpamb/cases/")) {
                    return null;
                }
                try {
                    ClassFile cf = ClassFile.of();
                    ClassModel model = cf.parse(classfileBuffer);
                    String internalClassName = model.thisClass().asInternalName();
                    return cf.transformClass(model, instrumentClass(internalClassName));
                } catch (Throwable t) {
                    System.err.println("[InstrumentationAgent] Failed transforming " + className + ": " + t);
                    return null;
                }
            }
        });
    }

    public static void recordSite(String methodId, int pcOffset) {
        VISITED_SITES.add(methodId + "#" + pcOffset);
    }

    public static void recordIntValue(String methodId, int pcOffset, int position, int value) {
        String key = methodId + "#" + pcOffset + "#" + position;
        OBSERVED_VALUES.computeIfAbsent(key, k -> new CopyOnWriteArrayList<>()).add(value);
    }

    public static void recordOutcome(String outcome) {
        if (outcome != null && !outcome.isBlank()) {
            OBSERVED_OUTCOMES.add(outcome.trim());
        }
    }

    public static void recordThrowable(Throwable t) {
        if (t == null) {
            recordOutcome("ok");
            return;
        }
        Throwable cur = t;
        while (cur != null) {
            if (cur instanceof AssertionError) {
                recordOutcome("assertion error");
                return;
            }
            if (cur instanceof IndexOutOfBoundsException) {
                recordOutcome("out of bounds");
                return;
            }
            if (cur instanceof ArithmeticException) {
                recordOutcome("divide by zero");
                return;
            }
            if (cur instanceof NullPointerException) {
                recordOutcome("null pointer");
                return;
            }
            cur = cur.getCause();
        }
        recordOutcome("*");
    }

    public static String dumpSummary() {
        Map<String, List<Integer>> snap = new LinkedHashMap<>();
        OBSERVED_VALUES.keySet().stream().sorted().forEach(k -> {
            CopyOnWriteArrayList<Integer> list = OBSERVED_VALUES.get(k);
            if (list != null && !list.isEmpty()) {
                snap.put(k, new ArrayList<>(list));
            }
        });
        return "AGENT_SITES:" + VISITED_SITES.size() + " AGENT_VALUE_KEYS:" + snap.size();
    }

    private static ClassTransform instrumentClass(String internalClassName) {
        return (ClassBuilder cb, ClassElement element) -> {
            if (element instanceof MethodModel mm) {
                String methodId = formatMethodId(internalClassName, mm);
                cb.transformMethod(mm, MethodTransform.transformingCode(instrumentMethod(methodId)));
            } else {
                cb.with(element);
            }
        };
    }

    private static CodeTransform instrumentMethod(String methodId) {
        final int[] pcCounter = {0};
        return (CodeBuilder cob, CodeElement el) -> {
            if (!(el instanceof Instruction inst)) {
                cob.with(el);
                return;
            }

            int pcOffset = pcCounter[0]++;

            if (inst instanceof BranchInstruction br) {
                emitRecordSite(cob, methodId, pcOffset);
                if (isTwoOperandIntBranch(br.opcode())) {
                    emitTwoIntOperands(cob, methodId, pcOffset);
                } else if (isSingleOperandIntBranch(br.opcode())) {
                    emitTopIntOperand(cob, methodId, pcOffset, 0);
                }
            } else if (inst instanceof OperatorInstruction op && isDivOrRem(op)) {
                emitTwoIntOperands(cob, methodId, pcOffset);
            } else if (inst instanceof ArrayLoadInstruction al && isSingleWordIntArrayLoad(al.opcode())) {
                emitTopIntOperand(cob, methodId, pcOffset, 0);
            }

            cob.with(inst);
        };
    }

    private static boolean isSingleWordIntArrayLoad(Opcode op) {
        return switch (op) {
            case IALOAD, BALOAD, CALOAD, SALOAD -> true;
            default -> false;
        };
    }

    private static boolean isSingleOperandIntBranch(Opcode op) {
        return switch (op) {
            case IFEQ, IFNE, IFLT, IFGE, IFGT, IFLE -> true;
            default -> false;
        };
    }

    private static boolean isTwoOperandIntBranch(Opcode op) {
        return switch (op) {
            case IF_ICMPEQ, IF_ICMPNE, IF_ICMPLT, IF_ICMPGE, IF_ICMPGT, IF_ICMPLE -> true;
            default -> false;
        };
    }

    private static boolean isDivOrRem(OperatorInstruction op) {
        return switch (op.opcode()) {
            case IDIV, IREM -> true;
            default -> false;
        };
    }

    private static void emitRecordSite(CodeBuilder cob, String methodId, int pcOffset) {
        cob.ldc(methodId);
        cob.loadConstant(pcOffset);
        cob.invokestatic(AGENT_DESC, "recordSite", RECORD_SITE_DESC, false);
    }

    // Stack before: [..., topInt] -> leaves [..., topInt]
    private static void emitTopIntOperand(
            CodeBuilder cob, String methodId, int pcOffset, int position) {
        int topLocal = cob.allocateLocal(TypeKind.INT);
        cob.dup();
        cob.istore(topLocal);
        cob.ldc(methodId);
        cob.loadConstant(pcOffset);
        cob.loadConstant(position);
        cob.iload(topLocal);
        cob.invokestatic(AGENT_DESC, "recordIntValue", RECORD_VAL_DESC, false);
    }

    // Stack before: [..., firstInt, secondInt] -> leaves [..., firstInt, secondInt]
    private static void emitTwoIntOperands(CodeBuilder cob, String methodId, int pcOffset) {
        int secondLocal = cob.allocateLocal(TypeKind.INT);
        int firstLocal = cob.allocateLocal(TypeKind.INT);
        cob.istore(secondLocal);
        cob.istore(firstLocal);

        cob.ldc(methodId);
        cob.loadConstant(pcOffset);
        cob.loadConstant(0);
        cob.iload(firstLocal);
        cob.invokestatic(AGENT_DESC, "recordIntValue", RECORD_VAL_DESC, false);

        cob.ldc(methodId);
        cob.loadConstant(pcOffset);
        cob.loadConstant(1);
        cob.iload(secondLocal);
        cob.invokestatic(AGENT_DESC, "recordIntValue", RECORD_VAL_DESC, false);

        cob.iload(firstLocal);
        cob.iload(secondLocal);
    }

    private static String formatMethodId(String internalClassName, MethodModel mm) {
        String dottedClass = internalClassName.replace('/', '.');
        String name = mm.methodName().stringValue();
        String desc = mm.methodType().stringValue();
        return dottedClass + "." + name + ":" + desc;
    }

    private static synchronized void writeOutput(Path target) {
        try {
            if (target.getParent() != null) {
                Files.createDirectories(target.getParent());
            }
            List<String> sites = new ArrayList<>(VISITED_SITES);
            sites.sort(String::compareTo);

            List<String> keys = new ArrayList<>(OBSERVED_VALUES.keySet());
            keys.sort(String::compareTo);

            List<String> outcomes = new ArrayList<>(OBSERVED_OUTCOMES);
            outcomes.sort(String::compareTo);

            try (BufferedWriter writer = Files.newBufferedWriter(target, StandardCharsets.UTF_8)) {
                writer.write("SITES\n");
                for (String s : sites) {
                    writer.write(s + "\n");
                }
                writer.write("VALUES\n");
                for (String k : keys) {
                    CopyOnWriteArrayList<Integer> vals = OBSERVED_VALUES.get(k);
                    if (vals != null && !vals.isEmpty()) {
                        writer.write(k + "=");
                        for (int i = 0; i < vals.size(); i++) {
                            if (i > 0) {
                                writer.write(",");
                            }
                            writer.write(Integer.toString(vals.get(i)));
                        }
                        writer.write("\n");
                    }
                }
                writer.write("OUTCOMES\n");
                for (String o : outcomes) {
                    writer.write(o + "\n");
                }
            }
        } catch (IOException e) {
            System.err.println("[InstrumentationAgent] Failed writing output to " + target + ": " + e);
        }
    }
}
