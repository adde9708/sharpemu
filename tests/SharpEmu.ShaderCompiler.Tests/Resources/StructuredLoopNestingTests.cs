// Copyright (C) 2026 SharpEmu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

using SharpEmu.ShaderCompiler.Vulkan;
using Xunit;
using static SharpEmu.ShaderCompiler.Tests.Resources.ResourceTestProgram;

namespace SharpEmu.ShaderCompiler.Tests.Resources;

// A natural loop is headed by a block that a backward branch targets, and ends at the last block
// branching back to it. Structured control flow can only express these when the regions nest: an
// inner loop whose latch reaches past the latch of the loop containing it leaves the two
// interleaved, and that program keeps the block dispatcher instead.
//
// The block dispatcher switches on the program counter; the structured path emits loops and no
// such switch. OpLoopMerge cannot tell the two apart because the dispatcher opens a loop of its
// own, so the switch is what decides which path a program took. A branch target is
// pc + 4 + offset * 4.
public sealed class StructuredLoopNestingTests
{
    // Every branch targets the next pc, so each instruction starts a block. The branch at pc 12
    // reaches back to pc 0, so the loop headed at block 0 latches at block 3; the one at pc 8
    // reaches back to pc 4, so the loop headed at block 1 latches at block 2, inside the outer.
    private static Gen5ShaderProgram ProperlyNested() => Program(
        Branch(0, "SBranch", 0),
        Branch(4, "SBranch", 0),
        Branch(8, "SBranch", -2),
        Branch(12, "SBranch", -4),
        EndProgram(16));

    // The same shape, except the branch at pc 12 reaches back to pc 4 instead of pc 0. That makes
    // the regions {0->2} and {2->3}, which cross: the inner loop latches past the outer one.
    private static Gen5ShaderProgram Interleaved() => Program(
        Branch(0, "SBranch", 0),
        Branch(4, "SBranch", 0),
        Branch(8, "SBranch", -3),
        Branch(12, "SBranch", -2),
        EndProgram(16));

    // A single loop: the backward branch at pc 8 targets pc 0.
    private static Gen5ShaderProgram SingleLoop() => Program(
        MoveScalar(0, 12, 1),
        MoveScalar(4, 13, 1),
        Branch(8, "SBranch", -3),
        EndProgram(12));

    // No backward branch at all, so there is no loop region to speak of.
    private static Gen5ShaderProgram LoopFree() => Program(
        MoveScalar(0, 12, 1),
        MoveScalar(4, 13, 1),
        EndProgram(8));

    // Returns the number of OpSwitch instructions, which the block dispatcher emits per
    // non-structured block and the structured path never emits.
    private static int Inspect(Gen5ShaderProgram program)
    {
        Assert.True(Gen5SpirvTranslator.TryCompileProgram(Request(program), out var shader, out var error), error);
        return new SpirvModuleInspector(shader.Spirv).Opcodes.Count(opcode => (SpirvOp)opcode == SpirvOp.Switch);
    }

    [Fact]
    public void ProperlyNestedLoopsUseTheStructuredPath()
    {
        Assert.Equal(0, Inspect(ProperlyNested()));
    }

    [Fact]
    public void InterleavedLoopsFallBackToTheBlockDispatcher()
    {
        // The two regions cross, so structured control flow cannot express them.
        Assert.True(Inspect(Interleaved()) > 0);
    }

    [Fact]
    public void ASingleNaturalLoopUsesTheStructuredPath()
    {
        Assert.Equal(0, Inspect(SingleLoop()));
    }

    [Fact]
    public void ALoopFreeProgramUsesTheStructuredPath()
    {
        Assert.Equal(0, Inspect(LoopFree()));
    }

    [Theory]
    [InlineData("nested")]
    [InlineData("interleaved")]
    [InlineData("single")]
    [InlineData("loopfree")]
    public void EveryLoopShapeStillCompiles(string shape)
    {
        var program = shape switch
        {
            "nested" => ProperlyNested(),
            "interleaved" => Interleaved(),
            "single" => SingleLoop(),
            _ => LoopFree(),
        };

        // Rejecting the structured path is a valid outcome, so no shape may fail the compile.
        Assert.True(Gen5SpirvTranslator.TryCompileProgram(Request(program), out var shader, out var error), error);
        Assert.NotEmpty(shader.Spirv);
    }
}
