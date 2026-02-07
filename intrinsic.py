import numba
from numba import types
from numba.extending import intrinsic
from numba.core import cgutils

def llvm_intrinsic(name, has_is_zero_undef=False):
    """
    Factory to create a Numba intrinsic for an LLVM integer intrinsic.
    has_is_zero_undef: True for ctlz/cttz which require an extra bool argument.
    """
    @intrinsic
    def intrinsic_impl(typingctx, src):
        if not isinstance(src, types.Integer):
            raise TypeError(f"{name} intrinsic only supports integer types")
        bit_width = src.bitwidth
        llvm_name = f"llvm.{name}.i{bit_width}"

        sig = src(src)

        def codegen(context, builder, signature, args):
            val = args[0]
            module = builder.module
            if has_is_zero_undef:
                bool_t = cgutils.ir.IntType(1)
                fnty = cgutils.ir.FunctionType(val.type, [val.type, bool_t])
                fn = cgutils.get_or_insert_function(module, fnty, llvm_name)
                return builder.call(fn, [val, cgutils.false_bit])  # safe if input is zero
            else:
                fnty = cgutils.ir.FunctionType(val.type, [val.type])
                fn = cgutils.get_or_insert_function(module, fnty, llvm_name)
                return builder.call(fn, [val])

        return sig, codegen

    return intrinsic_impl

# --- Create the actual functions ---
native_popcount = llvm_intrinsic("ctpop")
native_clz = llvm_intrinsic("ctlz", has_is_zero_undef=False)
native_ctz = llvm_intrinsic("cttz", has_is_zero_undef=False)

# --- Easy-to-use wrappers ---
@numba.njit(inline='always')
def popcount(x):
    return native_popcount(x)

@numba.njit(inline='always')
def clz(x):
    return native_clz(x)

@numba.njit(inline='always')
def ctz(x):
    return native_ctz(x)