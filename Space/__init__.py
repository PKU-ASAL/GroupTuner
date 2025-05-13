# initialize function of options
option_function = {
    '-finline-atomics': False,
    '-fcx-limited-range': False,
    '-fstdarg-opt': False,
    '-freg-struct-return': False,
    '-fstrict-volatile-bitfields': False,
    '-fshort-wchar': False,
    '-ftree-switch-conversion': False, # 需要开启该flag才能测试到
    '-ftree-builtin-call-dce': False,
    '-fsplit-wide-types': False,
    '-foptimize-strlen': False,
    # '-finline-small-functions': False,
    '-fcx-fortran-rules': False,
    '-fdevirtualize': False,
    '-fdevirtualize-speculatively': False,
    '-fshort-enums': False,
    '-fprintf-return-value': False,
    '-fjump-tables': False,
    '-ffp-int-builtin-inexact': False,

    '-fpcc-struct-return': False
}