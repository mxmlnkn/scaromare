__device__ void MonteCarloPiKernel_gpuMethod0_(int thisref, int * exception){
    int       r0   =-1;
    int       i0   = 0;
    int       i1   = 0;
    long long l2   = 0;
    int       i3   = 0;
    float     f0   = 0;
    float     f1   = 0;
    long long $l4  = 0;
    char      $z0  = 0;
    long long $l5  = 0;
    char      $b6  = 0;
    int       $r1  =-1;
    long long $l7  = 0;
    long long $l8  = 0;
    long long $l9  = 0;
    long long $l10 = 0;
    int       i11  = 0;
    float     $f2  = 0;
    long long $l12 = 0;
    long long $l13 = 0;
    long long $l14 = 0;
    float     $f3  = 0;
    float     $f4  = 0;
    float     $f5  = 0;
    float     $f6  = 0;
    double    $d0  = 0;
    char      $b15 = 0;
    int       $r2  =-1;
    int       $i16 = 0;

    r0  = thisref;
    $l4 = instance_getter_MonteCarloPiKernel_mRandomSeed(r0, exception);
    if(*exception != 0)
        return;
    i0 = (int) $l4;
    $z0 = static_getter_MonteCarloPiKernel_$assertionsDisabled(exception);
    if ( $z0 != 0 ) goto label0;
    $l5 = instance_getter_MonteCarloPiKernel_mnDiceRolls(r0, exception);
    if(*exception != 0)
        return;
    $b6 = org_trifort_cmp( $l5 , 2147483647L );
    if ( $b6 <= 0 ) goto label0;
    $r1 = -1;
    org_trifort_gc_assign (
    &$r1, java_lang_AssertionError_initab850b60f96d11de8a390800200c9a660_(exception));
    if(*exception != 0)
        return;
    *exception = $r1;
    return;

label0:
    $l7 = instance_getter_MonteCarloPiKernel_mnDiceRolls(r0, exception);
    if(*exception != 0)
        return;
    i1 = (int) $l7;
    l2 = 0L;
    i3 = 0;
    label4:
    if ( i3 >= i1 )
        goto label2;
    $l8  = (long long) i0;
    $l9  = 950706376L * $l8;
    $l10 = $l9 % 2147483647L;
    i11  = (int) $l10;
    $f2  = (float) i11;
    f0   = $f2 / 2.14748365E9F;
    $l12 = (long long) i11;
    $l13 = 950706376L * $l12;
    $l14 = $l13 % 2147483647L;
    i0   = (int) $l14;
    $f3  = (float) i0;
    f1   = $f3 / 2.14748365E9F;
    $f4  = f0 * f0;
    $f5  = f1 * f1;
    $f6  = $f4 + $f5;
    $d0  = (double) $f6;
    $b15 = org_trifort_cmpg((double) $d0 , (double) 1.0 );
    if ( $b15 >= 0 )
        goto label3;
    l2 = l2 + 1L ;
label3:
    i3 = i3 + 1 ;
    goto label4;

label2:
    $r2 = instance_getter_MonteCarloPiKernel_mnHits(r0, exception);
    if(*exception != 0)
        return;
    $i16 = instance_getter_MonteCarloPiKernel_miLinearThreadId(r0, exception);
    if(*exception != 0)
        return;
    long__array_set($r2, $i16, l2 , exception);
    if(*exception != 0)
        return;

    return;
}
