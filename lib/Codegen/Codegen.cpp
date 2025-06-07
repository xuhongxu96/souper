// Copyright 2014 The Souper Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "souper/Codegen/Codegen.h"
#include "souper/Inst/Inst.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include <map>
#include <queue>
#include <string>

#define DEBUG_TYPE "souper"
STATISTIC(InstructionReplaced,
          "Number of instructions replaced by another instruction");
STATISTIC(DominanceCheckFailed,
          "Number of failed replacement due to dominance check");

using namespace llvm;

extern unsigned DebugLevel;

namespace souper {

llvm::Type *Codegen::GetInstReturnType(llvm::LLVMContext &Context, Inst *I) {
  switch (I->K) {
  case Inst::SAddWithOverflow:
  case Inst::UAddWithOverflow:
  case Inst::SSubWithOverflow:
  case Inst::USubWithOverflow:
  case Inst::SMulWithOverflow:
  case Inst::UMulWithOverflow:
    return StructType::get(Context, {Type::getIntNTy(Context, I->Width - 1),
                                     Type::getInt1Ty(Context)});
  default:
    return Type::getIntNTy(Context, I->Width);
  }
}

llvm::Value *Codegen::getValue(Inst *I) {
  const std::vector<Inst *> &Ops = I->orderedOps();
  if (I->K == Inst::UntypedConst) {
    // FIXME: We only get here because it is the second argument of
    // extractvalue instrs. This is not otherwise reachable.
    // BUT, we can't return nullptr, else we completely bailout.
    return llvm::ConstantInt::getFalse(Context);
  }

  Type *T;
  if (I->K != Inst::ExtractValue) // it depends on the second value then
    T = Type::getIntNTy(Context, I->Width);

  if (I->K == Inst::Const)
    return ConstantInt::get(T, I->Val);

  if (ReplacedValues.find(I) != ReplacedValues.end())
    return ReplacedValues.at(I);

  if (InstValueMap) {
    if (auto it = InstValueMap->find(I); it != InstValueMap->end()) {
      return it->second;
    }
  }

  auto SaveToInstValueMap = [this, I](Value *V) {
    if (InstValueMap) {
      (*InstValueMap)[I] = V;
    }
    return V;
  };

  if (I->Origins.size() > 0) {
    // if there's an Origin, we're connecting to existing code
    for (auto V : I->Origins) {
      if (V->getType() != T)
        continue; // TODO: can we assert this doesn't happen?
      if (isa<Argument>(V) || isa<Constant>(V))
        return SaveToInstValueMap(V);
      if (auto IP = dyn_cast<Instruction>(V)) {
        if (DT->dominates(IP, ReplacedInst)) {
          ++InstructionReplaced;
          return SaveToInstValueMap(V);
        } else {
          if (DebugLevel > 2)
            llvm::errs() << "dominance check failed\n";
          ++DominanceCheckFailed;
        }
      } else {
        report_fatal_error("Unhandled LLVM instruction in getValue()");
      }
    }
    if (DebugLevel > 2)
      llvm::errs() << "returning nullptr from getValue()\n";
    return SaveToInstValueMap(nullptr);
  }

  // otherwise, recursively generate code
  Value *V0 = Codegen::getValue(Ops[0]);
  if (!V0)
    return SaveToInstValueMap(nullptr);

  BasicBlock *ForcedInsertPoint = nullptr;
  if (auto it = InstToBlockMap.find(I); it != InstToBlockMap.end()) {
    ForcedInsertPoint = it->second;
  }

  // PHI nodes must be the first instructions in a basic block. If we're
  // replacing a PHI node with another instruction, make sure it comes after the
  // other PHI nodes.
  if (ReplacedInst) {
    Instruction *InsertPoint = ReplacedInst;
    while (isa<PHINode>(InsertPoint->getNextNode()))
      InsertPoint = InsertPoint->getNextNode();
    Builder.SetInsertPoint(InsertPoint->getNextNode());
  }

  if (ForcedInsertPoint) {
    Builder.SetInsertPoint(ForcedInsertPoint);
  }

  switch (Ops.size()) {
  case 1: {
    switch (I->K) {
    case Inst::SExt:
      return SaveToInstValueMap(Builder.CreateSExt(V0, T));
    case Inst::ZExt:
      return SaveToInstValueMap(Builder.CreateZExt(V0, T));
    case Inst::Trunc:
      return SaveToInstValueMap(Builder.CreateTrunc(V0, T));
    case Inst::CtPop: {
      Function *F = Intrinsic::getDeclaration(M, Intrinsic::ctpop, T);
      return SaveToInstValueMap(Builder.CreateCall(F, V0));
    }
    case Inst::BSwap: {
      Function *F = Intrinsic::getDeclaration(M, Intrinsic::bswap, T);
      return SaveToInstValueMap(Builder.CreateCall(F, V0));
    }
    case Inst::BitReverse: {
      Function *F = Intrinsic::getDeclaration(M, Intrinsic::bitreverse, T);
      return SaveToInstValueMap(Builder.CreateCall(F, V0));
    }
    case Inst::Cttz: {
      Function *F = Intrinsic::getDeclaration(M, Intrinsic::cttz, T);
      // According to LLVM LangRef, the second argument of cttz i1
      // <is_zero_undef> must be a constant and is a flag to indicate whether
      // the intrinsic should ensure that a zero as the first argument produces
      // a defined result.
      return SaveToInstValueMap(Builder.CreateCall(
          F, {V0, ConstantInt::get(V0->getContext(), APInt(1, 0))}));
    }
    case Inst::Ctlz: {
      // Ditto
      Function *F = Intrinsic::getDeclaration(M, Intrinsic::ctlz, T);
      return SaveToInstValueMap(Builder.CreateCall(
          F, {V0, ConstantInt::get(V0->getContext(), APInt(1, 0))}));
    }
    case Inst::Freeze:
      return SaveToInstValueMap(Builder.CreateFreeze(V0));
    default:
      break;
    }
    break;
  }
  case 2: {
    Value *V1 = Codegen::getValue(Ops[1]);
    if (ForcedInsertPoint) {
      Builder.SetInsertPoint(ForcedInsertPoint);
    }
    if (!V1)
      return SaveToInstValueMap(nullptr);
    switch (I->K) {
    case Inst::And:
      return SaveToInstValueMap(Builder.CreateAnd(V0, V1));
    case Inst::Or:
      return SaveToInstValueMap(Builder.CreateOr(V0, V1));
    case Inst::Xor:
      return SaveToInstValueMap(Builder.CreateXor(V0, V1));
    case Inst::Add:
    case Inst::AddNSW:
    case Inst::AddNUW:
    case Inst::AddNW:
      return SaveToInstValueMap(Builder.CreateAdd(
          V0, V1, /*Name=*/{},
          /*HasNUW=*/I->K == Inst::AddNW || I->K == Inst::AddNUW,
          /*HasNSW=*/I->K == Inst::AddNW || I->K == Inst::AddNSW));
    case Inst::Sub:
    case Inst::SubNSW:
    case Inst::SubNUW:
    case Inst::SubNW:
      return SaveToInstValueMap(Builder.CreateSub(
          V0, V1, /*Name=*/{},
          /*HasNUW=*/I->K == Inst::SubNW || I->K == Inst::SubNUW,
          /*HasNSW=*/I->K == Inst::SubNW || I->K == Inst::SubNSW));
    case Inst::Mul:
    case Inst::MulNSW:
    case Inst::MulNUW:
    case Inst::MulNW:
      return SaveToInstValueMap(Builder.CreateMul(
          V0, V1, /*Name=*/{},
          /*HasNUW=*/I->K == Inst::MulNW || I->K == Inst::MulNUW,
          /*HasNSW=*/I->K == Inst::MulNW || I->K == Inst::MulNSW));
    case Inst::UDiv:
    case Inst::UDivExact:
      return SaveToInstValueMap(
          Builder.CreateUDiv(V0, V1, /*Name=*/{},
                             /*IsExact=*/I->K == Inst::UDivExact));
    case Inst::SDiv:
    case Inst::SDivExact:
      return SaveToInstValueMap(
          Builder.CreateSDiv(V0, V1, /*Name=*/{},
                             /*IsExact=*/I->K == Inst::SDivExact));
    case Inst::URem:
      return SaveToInstValueMap(Builder.CreateURem(V0, V1));
    case Inst::SRem:
      return SaveToInstValueMap(Builder.CreateSRem(V0, V1));
    case Inst::Shl:
    case Inst::ShlNSW:
    case Inst::ShlNUW:
    case Inst::ShlNW:
      return SaveToInstValueMap(Builder.CreateShl(
          V0, V1, /*Name=*/{},
          /*HasNUW=*/I->K == Inst::ShlNW || I->K == Inst::ShlNUW,
          /*HasNSW=*/I->K == Inst::ShlNW || I->K == Inst::ShlNSW));
    case Inst::AShr:
    case Inst::AShrExact:
      return SaveToInstValueMap(
          Builder.CreateAShr(V0, V1, /*Name=*/{},
                             /*IsExact=*/I->K == Inst::AShrExact));
    case Inst::LShr:
    case Inst::LShrExact:
      return SaveToInstValueMap(
          Builder.CreateLShr(V0, V1, /*Name=*/{},
                             /*IsExact=*/I->K == Inst::LShrExact));
    case Inst::Ne:
      return SaveToInstValueMap(Builder.CreateICmpNE(V0, V1));
    case Inst::Eq:
      return SaveToInstValueMap(Builder.CreateICmpEQ(V0, V1));
    case Inst::Ult:
      return SaveToInstValueMap(Builder.CreateICmpULT(V0, V1));
    case Inst::Slt:
      return SaveToInstValueMap(Builder.CreateICmpSLT(V0, V1));
    case Inst::Ule:
      return SaveToInstValueMap(Builder.CreateICmpULE(V0, V1));
    case Inst::Sle:
      return SaveToInstValueMap(Builder.CreateICmpSLE(V0, V1));
    case Inst::SAddO:
    case Inst::UAddO:
    case Inst::SSubO:
    case Inst::USubO:
    case Inst::SMulO:
    case Inst::UMulO:
      // FIXME: We only get here because it is the second argument of
      // ".with.overflow" instrs. This is not otherwise reachable.
      // BUT, we can't return SaveToInstValueMap(nullptr, else we completely
      // bailout.
      return SaveToInstValueMap(llvm::ConstantInt::getFalse(Context));
    case Inst::ExtractValue:
      return SaveToInstValueMap(
          Builder.CreateExtractValue(V0, I->Ops[1]->Val.getZExtValue()));
    case Inst::SAddWithOverflow:
    case Inst::UAddWithOverflow:
    case Inst::SSubWithOverflow:
    case Inst::USubWithOverflow:
    case Inst::SMulWithOverflow:
    case Inst::UMulWithOverflow: {
      if (Ops[0]->Ops != Ops[1]->Ops) {
        report_fatal_error(
            "Inst::*WithOverflow with non-identical args unsupported.");
      }
      V0 = Codegen::getValue(Ops[0]->orderedOps()[0]);
      V1 = Codegen::getValue(Ops[0]->orderedOps()[1]);
      if (ForcedInsertPoint) {
        Builder.SetInsertPoint(ForcedInsertPoint);
      }
      Intrinsic::ID ID = [K = I->K]() {
        switch (K) {
        case Inst::SAddWithOverflow:
          return Intrinsic::sadd_with_overflow;
        case Inst::UAddWithOverflow:
          return Intrinsic::uadd_with_overflow;
        case Inst::SSubWithOverflow:
          return Intrinsic::ssub_with_overflow;
        case Inst::USubWithOverflow:
          return Intrinsic::usub_with_overflow;
        case Inst::SMulWithOverflow:
          return Intrinsic::smul_with_overflow;
        case Inst::UMulWithOverflow:
          return Intrinsic::umul_with_overflow;
        default:
          break;
        };
        report_fatal_error("Unexpected overflow inst");
      }();
      T = Type::getIntNTy(Context, Ops[0]->orderedOps()[0]->Width);
      Function *F = Intrinsic::getDeclaration(M, ID, T);
      return SaveToInstValueMap(Builder.CreateCall(F, {V0, V1}));
    }
    case Inst::SAddSat:
      return SaveToInstValueMap(Builder.CreateCall(
          Intrinsic::getDeclaration(M, Intrinsic::sadd_sat, T), {V0, V1}));
    case Inst::UAddSat:
      return SaveToInstValueMap(Builder.CreateCall(
          Intrinsic::getDeclaration(M, Intrinsic::uadd_sat, T), {V0, V1}));
    case Inst::SSubSat:
      return SaveToInstValueMap(Builder.CreateCall(
          Intrinsic::getDeclaration(M, Intrinsic::ssub_sat, T), {V0, V1}));
    case Inst::USubSat:
      return SaveToInstValueMap(Builder.CreateCall(
          Intrinsic::getDeclaration(M, Intrinsic::usub_sat, T), {V0, V1}));
    default:
      break;
    }
    break;
  }
  case 3: {
    Value *V1 = Codegen::getValue(Ops[1]);
    Value *V2 = Codegen::getValue(Ops[2]);
    if (ForcedInsertPoint) {
      Builder.SetInsertPoint(ForcedInsertPoint);
    }
    if (!V1 || !V2)
      return SaveToInstValueMap(nullptr);
    switch (I->K) {
    case Inst::Select:
      return SaveToInstValueMap(Builder.CreateSelect(V0, V1, V2));
    case Inst::FShl:
    case Inst::FShr: {
      Intrinsic::ID ID = I->K == Inst::FShl ? Intrinsic::fshl : Intrinsic::fshr;
      Function *F = Intrinsic::getDeclaration(M, ID, T);
      return SaveToInstValueMap(Builder.CreateCall(F, {V0, V1, V2}));
    }
    default:
      break;
    }
    break;
  }
  default:
    break;
  }

  // FIXME: ExtractValue
  // FIXME: @llvm.[us]{add,sub,mul}.with.overflow (two next ones on same args)
  // FIXME: [US]{Add,Sub,Mul}WithOverflow
  // FIXME: [US]{Add,Sub,Mul}O
  // FIXME: PHI

  report_fatal_error(((std::string) "Unhandled Souper instruction " +
                      Inst::getKindName(I->K) + " in Codegen::getValue()")
                         .c_str());
}

static std::vector<llvm::Type *>
GetInputArgumentTypes(const InstContext &IC, llvm::LLVMContext &Context,
                      Inst *Root) {
  const std::vector<Inst *> AllVariables = IC.getVariablesFor(Root);

  std::vector<llvm::Type *> ArgTypes;
  ArgTypes.reserve(AllVariables.size());
  for (const Inst *const Var : AllVariables) {
    // llvm::errs() << "arg with width " << Var->Width << " and number " <<
    // Var->Number << "\n";
    ArgTypes.emplace_back(Type::getIntNTy(Context, Var->Width));
  }

  return ArgTypes;
}

static std::map<Inst *, Value *> GetArgsMapping(const InstContext &IC,
                                                Function *F, Inst *Root) {
  std::map<Inst *, Value *> Args;

  const std::vector<Inst *> AllVariables = IC.getVariablesFor(Root);
  for (auto zz : llvm::zip(AllVariables, F->args()))
    Args[std::get<0>(zz)] = &(std::get<1>(zz));

  return Args;
};

static std::vector<llvm::Type *>
GetInputArgumentTypes(const InstContext &IC, llvm::LLVMContext &Context) {
  const std::vector<Inst *> AllVariables = IC.getVariables();

  std::vector<llvm::Type *> ArgTypes;
  ArgTypes.reserve(AllVariables.size());
  for (const Inst *const Var : AllVariables) {
    // llvm::errs() << "arg with width " << Var->Width << " and number " <<
    // Var->Number << "\n";
    ArgTypes.emplace_back(Type::getIntNTy(Context, Var->Width));
  }

  return ArgTypes;
}

static std::map<Inst *, Value *> GetArgsMapping(const InstContext &IC,
                                                Function *F) {
  std::map<Inst *, Value *> Args;

  const std::vector<Inst *> AllVariables = IC.getVariables();
  for (auto zz : llvm::zip(AllVariables, F->args()))
    Args[std::get<0>(zz)] = &(std::get<1>(zz));

  return Args;
};

/// If there are no errors, the function returns false. If an error is found,
/// a message describing the error is written to OS (if non-null) and true is
/// returned.
bool genModule(InstContext &IC, souper::Inst *I, llvm::Module &Module) {
  llvm::LLVMContext &Context = Module.getContext();
  const std::vector<llvm::Type *> ArgTypes =
      GetInputArgumentTypes(IC, Context, I);
  const auto FT = llvm::FunctionType::get(
      /*Result=*/Codegen::GetInstReturnType(Context, I),
      /*Params=*/ArgTypes, /*isVarArg=*/false);

  Function *F = Function::Create(FT, Function::ExternalLinkage, "fun", &Module);

  const std::map<Inst *, Value *> Args = GetArgsMapping(IC, F, I);

  BasicBlock *BB = BasicBlock::Create(Context, "entry", F);

  llvm::IRBuilder<> Builder(Context);
  Builder.SetInsertPoint(BB);

  Value *RetVal = Codegen(Context, &Module, Builder, /*DT*/ nullptr,
                          /*ReplacedInst*/ nullptr, Args)
                      .getValue(I);

  Builder.CreateRet(RetVal);

  // Validate the generated code, checking for consistency.
  if (verifyFunction(*F, &llvm::errs()))
    return true;
  if (verifyModule(Module, &llvm::errs()))
    return true;
  return false;
}

bool genModuleWithBranches(InstContext &IC, Inst *I,
                           const std::vector<InstMapping> &PCs,
                           llvm::Module &Module) {
  llvm::LLVMContext &Context = Module.getContext();
  std::vector<llvm::Type *> ArgTypes = GetInputArgumentTypes(IC, Context);
  const auto FT = llvm::FunctionType::get(
      /*Result=*/Codegen::GetInstReturnType(Context, I),
      /*Params=*/ArgTypes, /*isVarArg=*/false);

  Function *F = Function::Create(FT, Function::ExternalLinkage, "fun", &Module);

  const std::map<Inst *, Value *> Args = GetArgsMapping(IC, F);

  llvm::IRBuilder<> Builder(Context);

  // Create basic blocks
  BasicBlock *BB_Entry = BasicBlock::Create(Context, "entry", F);
  std::vector<BasicBlock *> BB_PCs;
  for (int i = 0; i < PCs.size(); ++i) {
    BB_PCs.push_back(BasicBlock::Create(Context, "pc_" + std::to_string(i), F));
  }
  BasicBlock *BB_Unreachable =
      BB_PCs.empty() ? nullptr
                     : BasicBlock::Create(Context, "unreachable_block", F);

  // Map inst to blocks

  std::map<Inst *, BasicBlock *> BlockPerInst;
  auto map_recursively = [&BlockPerInst](Inst *root, BasicBlock *block) {
    std::queue<Inst *> Q;
    if (BlockPerInst.find(root) == BlockPerInst.end()) {
      Q.push(root);
    }
    while (!Q.empty()) {
      Inst *InstToProcess = Q.front();
      Q.pop();
      BlockPerInst[InstToProcess] = block;
      for (auto Op : InstToProcess->Ops) {
        if (BlockPerInst.find(Op) == BlockPerInst.end()) {
          Q.push(Op);
        }
      }
    }
  };
  for (int i = 0; i < BB_PCs.size(); ++i) {
    auto &pc = PCs[i];
    map_recursively(pc.LHS, i == 0 ? BB_Entry : BB_PCs[i - 1]);
  }
  if (!BB_PCs.empty()) {
    map_recursively(I, BB_PCs.back());
  }

  // Insert instructions

  std::map<Inst *, llvm::Value *> OutInstValueMap;

  Builder.SetInsertPoint(BB_PCs.empty() ? BB_Entry : BB_PCs.back());
  Value *RetVal =
      Codegen(Context, &Module, Builder, /*DT*/ nullptr,
              /*ReplacedInst*/ nullptr, Args, BlockPerInst, &OutInstValueMap)
          .getValue(I);

  Builder.CreateRet(RetVal);

  for (int i = 0; i < BB_PCs.size(); ++i) {
    auto &pc = PCs[i];
    Builder.SetInsertPoint(i == 0 ? BB_Entry : BB_PCs[i - 1]);

    if (pc.RHS->K != Inst::Const) {
      exit(-1);
    }

    Value *CondVal = nullptr;

    if (pc.RHS->Width != 1) {
      CondVal = Builder.CreateICmpEQ(
          Codegen(Context, &Module, Builder, /*DT*/ nullptr,
                  /*ReplacedInst*/ nullptr, Args, BlockPerInst,
                  &OutInstValueMap)
              .getValue(pc.LHS),
          Codegen(Context, &Module, Builder, /*DT*/ nullptr,
                  /*ReplacedInst*/ nullptr, Args, BlockPerInst,
                  &OutInstValueMap)
              .getValue(pc.RHS));
    } else {
      CondVal = Codegen(Context, &Module, Builder, /*DT*/ nullptr,
                        /*ReplacedInst*/ nullptr, Args, BlockPerInst,
                        &OutInstValueMap)
                    .getValue(pc.LHS);
    }

    BasicBlock *BB_True, *BB_False;
    if (pc.RHS->Val.getBoolValue()) {
      BB_True = BB_PCs[i];
      BB_False = BB_Unreachable;
    } else {
      BB_True = BB_Unreachable;
      BB_False = BB_PCs[i];
    }
    Builder.CreateCondBr(CondVal, BB_True, BB_False);
  }

  if (BB_Unreachable) {
    Builder.SetInsertPoint(BB_Unreachable);
    Builder.CreateUnreachable();
  }

  // std::error_code EC;
  // llvm::raw_fd_ostream OS("tmp", EC);
  // OS << Module;
  // OS.flush();

  // Validate the generated code, checking for consistency.
  if (verifyFunction(*F, &llvm::errs()))
    return true;
  if (verifyModule(Module, &llvm::errs()))
    return true;
  return false;
}

} // namespace souper
