::core::arch::global_asm!{ include_str!(concat!(env!("OUT_DIR"), "/asm.s")) }
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f32_simd512_m48_n1"]
                    unsafe fn __decl_f32_simd512x8_0__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f32_simd512_m48_n2"]
                    unsafe fn __decl_f32_simd512x8_1__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f32_simd512_m48_n3"]
                    unsafe fn __decl_f32_simd512x8_2__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f32_simd512_m48_n4"]
                    unsafe fn __decl_f32_simd512x8_3__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f32_simd512_m48_n5"]
                    unsafe fn __decl_f32_simd512x8_4__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f32_simd512_m48_n6"]
                    unsafe fn __decl_f32_simd512x8_5__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f32_simd512_m48_n7"]
                    unsafe fn __decl_f32_simd512x8_6__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f32_simd512_m48_n8"]
                    unsafe fn __decl_f32_simd512x8_7__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f32_simd512_m32_n1"]
                    unsafe fn __decl_f32_simd512x8_8__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f32_simd512_m32_n2"]
                    unsafe fn __decl_f32_simd512x8_9__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f32_simd512_m32_n3"]
                    unsafe fn __decl_f32_simd512x8_10__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f32_simd512_m32_n4"]
                    unsafe fn __decl_f32_simd512x8_11__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f32_simd512_m32_n5"]
                    unsafe fn __decl_f32_simd512x8_12__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f32_simd512_m32_n6"]
                    unsafe fn __decl_f32_simd512x8_13__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f32_simd512_m32_n7"]
                    unsafe fn __decl_f32_simd512x8_14__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f32_simd512_m32_n8"]
                    unsafe fn __decl_f32_simd512x8_15__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f32_simd512_m16_n1"]
                    unsafe fn __decl_f32_simd512x8_16__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f32_simd512_m16_n2"]
                    unsafe fn __decl_f32_simd512x8_17__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f32_simd512_m16_n3"]
                    unsafe fn __decl_f32_simd512x8_18__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f32_simd512_m16_n4"]
                    unsafe fn __decl_f32_simd512x8_19__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f32_simd512_m16_n5"]
                    unsafe fn __decl_f32_simd512x8_20__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f32_simd512_m16_n6"]
                    unsafe fn __decl_f32_simd512x8_21__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f32_simd512_m16_n7"]
                    unsafe fn __decl_f32_simd512x8_22__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f32_simd512_m16_n8"]
                    unsafe fn __decl_f32_simd512x8_23__();
                }
                pub static F32_SIMD512x8: [unsafe extern "C" fn(); 24] = [__decl_f32_simd512x8_0__,__decl_f32_simd512x8_1__,__decl_f32_simd512x8_2__,__decl_f32_simd512x8_3__,__decl_f32_simd512x8_4__,__decl_f32_simd512x8_5__,__decl_f32_simd512x8_6__,__decl_f32_simd512x8_7__,__decl_f32_simd512x8_8__,__decl_f32_simd512x8_9__,__decl_f32_simd512x8_10__,__decl_f32_simd512x8_11__,__decl_f32_simd512x8_12__,__decl_f32_simd512x8_13__,__decl_f32_simd512x8_14__,__decl_f32_simd512x8_15__,__decl_f32_simd512x8_16__,__decl_f32_simd512x8_17__,__decl_f32_simd512x8_18__,__decl_f32_simd512x8_19__,__decl_f32_simd512x8_20__,__decl_f32_simd512x8_21__,__decl_f32_simd512x8_22__,__decl_f32_simd512x8_23__,];
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c32_simd512_m24_n1"]
                    unsafe fn __decl_c32_simd512x8_0__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c32_simd512_m24_n2"]
                    unsafe fn __decl_c32_simd512x8_1__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c32_simd512_m24_n3"]
                    unsafe fn __decl_c32_simd512x8_2__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c32_simd512_m24_n4"]
                    unsafe fn __decl_c32_simd512x8_3__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c32_simd512_m24_n5"]
                    unsafe fn __decl_c32_simd512x8_4__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c32_simd512_m24_n6"]
                    unsafe fn __decl_c32_simd512x8_5__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c32_simd512_m24_n7"]
                    unsafe fn __decl_c32_simd512x8_6__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c32_simd512_m24_n8"]
                    unsafe fn __decl_c32_simd512x8_7__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c32_simd512_m16_n1"]
                    unsafe fn __decl_c32_simd512x8_8__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c32_simd512_m16_n2"]
                    unsafe fn __decl_c32_simd512x8_9__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c32_simd512_m16_n3"]
                    unsafe fn __decl_c32_simd512x8_10__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c32_simd512_m16_n4"]
                    unsafe fn __decl_c32_simd512x8_11__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c32_simd512_m16_n5"]
                    unsafe fn __decl_c32_simd512x8_12__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c32_simd512_m16_n6"]
                    unsafe fn __decl_c32_simd512x8_13__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c32_simd512_m16_n7"]
                    unsafe fn __decl_c32_simd512x8_14__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c32_simd512_m16_n8"]
                    unsafe fn __decl_c32_simd512x8_15__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c32_simd512_m8_n1"]
                    unsafe fn __decl_c32_simd512x8_16__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c32_simd512_m8_n2"]
                    unsafe fn __decl_c32_simd512x8_17__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c32_simd512_m8_n3"]
                    unsafe fn __decl_c32_simd512x8_18__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c32_simd512_m8_n4"]
                    unsafe fn __decl_c32_simd512x8_19__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c32_simd512_m8_n5"]
                    unsafe fn __decl_c32_simd512x8_20__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c32_simd512_m8_n6"]
                    unsafe fn __decl_c32_simd512x8_21__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c32_simd512_m8_n7"]
                    unsafe fn __decl_c32_simd512x8_22__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c32_simd512_m8_n8"]
                    unsafe fn __decl_c32_simd512x8_23__();
                }
                pub static C32_SIMD512x8: [unsafe extern "C" fn(); 24] = [__decl_c32_simd512x8_0__,__decl_c32_simd512x8_1__,__decl_c32_simd512x8_2__,__decl_c32_simd512x8_3__,__decl_c32_simd512x8_4__,__decl_c32_simd512x8_5__,__decl_c32_simd512x8_6__,__decl_c32_simd512x8_7__,__decl_c32_simd512x8_8__,__decl_c32_simd512x8_9__,__decl_c32_simd512x8_10__,__decl_c32_simd512x8_11__,__decl_c32_simd512x8_12__,__decl_c32_simd512x8_13__,__decl_c32_simd512x8_14__,__decl_c32_simd512x8_15__,__decl_c32_simd512x8_16__,__decl_c32_simd512x8_17__,__decl_c32_simd512x8_18__,__decl_c32_simd512x8_19__,__decl_c32_simd512x8_20__,__decl_c32_simd512x8_21__,__decl_c32_simd512x8_22__,__decl_c32_simd512x8_23__,];
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f64_simd512_m24_n1"]
                    unsafe fn __decl_f64_simd512x8_0__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f64_simd512_m24_n2"]
                    unsafe fn __decl_f64_simd512x8_1__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f64_simd512_m24_n3"]
                    unsafe fn __decl_f64_simd512x8_2__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f64_simd512_m24_n4"]
                    unsafe fn __decl_f64_simd512x8_3__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f64_simd512_m24_n5"]
                    unsafe fn __decl_f64_simd512x8_4__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f64_simd512_m24_n6"]
                    unsafe fn __decl_f64_simd512x8_5__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f64_simd512_m24_n7"]
                    unsafe fn __decl_f64_simd512x8_6__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f64_simd512_m24_n8"]
                    unsafe fn __decl_f64_simd512x8_7__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f64_simd512_m16_n1"]
                    unsafe fn __decl_f64_simd512x8_8__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f64_simd512_m16_n2"]
                    unsafe fn __decl_f64_simd512x8_9__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f64_simd512_m16_n3"]
                    unsafe fn __decl_f64_simd512x8_10__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f64_simd512_m16_n4"]
                    unsafe fn __decl_f64_simd512x8_11__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f64_simd512_m16_n5"]
                    unsafe fn __decl_f64_simd512x8_12__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f64_simd512_m16_n6"]
                    unsafe fn __decl_f64_simd512x8_13__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f64_simd512_m16_n7"]
                    unsafe fn __decl_f64_simd512x8_14__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f64_simd512_m16_n8"]
                    unsafe fn __decl_f64_simd512x8_15__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f64_simd512_m8_n1"]
                    unsafe fn __decl_f64_simd512x8_16__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f64_simd512_m8_n2"]
                    unsafe fn __decl_f64_simd512x8_17__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f64_simd512_m8_n3"]
                    unsafe fn __decl_f64_simd512x8_18__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f64_simd512_m8_n4"]
                    unsafe fn __decl_f64_simd512x8_19__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f64_simd512_m8_n5"]
                    unsafe fn __decl_f64_simd512x8_20__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f64_simd512_m8_n6"]
                    unsafe fn __decl_f64_simd512x8_21__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f64_simd512_m8_n7"]
                    unsafe fn __decl_f64_simd512x8_22__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f64_simd512_m8_n8"]
                    unsafe fn __decl_f64_simd512x8_23__();
                }
                pub static F64_SIMD512x8: [unsafe extern "C" fn(); 24] = [__decl_f64_simd512x8_0__,__decl_f64_simd512x8_1__,__decl_f64_simd512x8_2__,__decl_f64_simd512x8_3__,__decl_f64_simd512x8_4__,__decl_f64_simd512x8_5__,__decl_f64_simd512x8_6__,__decl_f64_simd512x8_7__,__decl_f64_simd512x8_8__,__decl_f64_simd512x8_9__,__decl_f64_simd512x8_10__,__decl_f64_simd512x8_11__,__decl_f64_simd512x8_12__,__decl_f64_simd512x8_13__,__decl_f64_simd512x8_14__,__decl_f64_simd512x8_15__,__decl_f64_simd512x8_16__,__decl_f64_simd512x8_17__,__decl_f64_simd512x8_18__,__decl_f64_simd512x8_19__,__decl_f64_simd512x8_20__,__decl_f64_simd512x8_21__,__decl_f64_simd512x8_22__,__decl_f64_simd512x8_23__,];
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c64_simd512_m12_n1"]
                    unsafe fn __decl_c64_simd512x8_0__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c64_simd512_m12_n2"]
                    unsafe fn __decl_c64_simd512x8_1__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c64_simd512_m12_n3"]
                    unsafe fn __decl_c64_simd512x8_2__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c64_simd512_m12_n4"]
                    unsafe fn __decl_c64_simd512x8_3__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c64_simd512_m12_n5"]
                    unsafe fn __decl_c64_simd512x8_4__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c64_simd512_m12_n6"]
                    unsafe fn __decl_c64_simd512x8_5__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c64_simd512_m12_n7"]
                    unsafe fn __decl_c64_simd512x8_6__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c64_simd512_m12_n8"]
                    unsafe fn __decl_c64_simd512x8_7__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c64_simd512_m8_n1"]
                    unsafe fn __decl_c64_simd512x8_8__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c64_simd512_m8_n2"]
                    unsafe fn __decl_c64_simd512x8_9__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c64_simd512_m8_n3"]
                    unsafe fn __decl_c64_simd512x8_10__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c64_simd512_m8_n4"]
                    unsafe fn __decl_c64_simd512x8_11__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c64_simd512_m8_n5"]
                    unsafe fn __decl_c64_simd512x8_12__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c64_simd512_m8_n6"]
                    unsafe fn __decl_c64_simd512x8_13__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c64_simd512_m8_n7"]
                    unsafe fn __decl_c64_simd512x8_14__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c64_simd512_m8_n8"]
                    unsafe fn __decl_c64_simd512x8_15__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c64_simd512_m4_n1"]
                    unsafe fn __decl_c64_simd512x8_16__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c64_simd512_m4_n2"]
                    unsafe fn __decl_c64_simd512x8_17__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c64_simd512_m4_n3"]
                    unsafe fn __decl_c64_simd512x8_18__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c64_simd512_m4_n4"]
                    unsafe fn __decl_c64_simd512x8_19__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c64_simd512_m4_n5"]
                    unsafe fn __decl_c64_simd512x8_20__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c64_simd512_m4_n6"]
                    unsafe fn __decl_c64_simd512x8_21__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c64_simd512_m4_n7"]
                    unsafe fn __decl_c64_simd512x8_22__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c64_simd512_m4_n8"]
                    unsafe fn __decl_c64_simd512x8_23__();
                }
                pub static C64_SIMD512x8: [unsafe extern "C" fn(); 24] = [__decl_c64_simd512x8_0__,__decl_c64_simd512x8_1__,__decl_c64_simd512x8_2__,__decl_c64_simd512x8_3__,__decl_c64_simd512x8_4__,__decl_c64_simd512x8_5__,__decl_c64_simd512x8_6__,__decl_c64_simd512x8_7__,__decl_c64_simd512x8_8__,__decl_c64_simd512x8_9__,__decl_c64_simd512x8_10__,__decl_c64_simd512x8_11__,__decl_c64_simd512x8_12__,__decl_c64_simd512x8_13__,__decl_c64_simd512x8_14__,__decl_c64_simd512x8_15__,__decl_c64_simd512x8_16__,__decl_c64_simd512x8_17__,__decl_c64_simd512x8_18__,__decl_c64_simd512x8_19__,__decl_c64_simd512x8_20__,__decl_c64_simd512x8_21__,__decl_c64_simd512x8_22__,__decl_c64_simd512x8_23__,];
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f32_simd512_m96_n1"]
                    unsafe fn __decl_f32_simd512x4_0__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f32_simd512_m96_n2"]
                    unsafe fn __decl_f32_simd512x4_1__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f32_simd512_m96_n3"]
                    unsafe fn __decl_f32_simd512x4_2__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f32_simd512_m96_n4"]
                    unsafe fn __decl_f32_simd512x4_3__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f32_simd512_m80_n1"]
                    unsafe fn __decl_f32_simd512x4_4__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f32_simd512_m80_n2"]
                    unsafe fn __decl_f32_simd512x4_5__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f32_simd512_m80_n3"]
                    unsafe fn __decl_f32_simd512x4_6__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f32_simd512_m80_n4"]
                    unsafe fn __decl_f32_simd512x4_7__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f32_simd512_m64_n1"]
                    unsafe fn __decl_f32_simd512x4_8__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f32_simd512_m64_n2"]
                    unsafe fn __decl_f32_simd512x4_9__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f32_simd512_m64_n3"]
                    unsafe fn __decl_f32_simd512x4_10__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f32_simd512_m64_n4"]
                    unsafe fn __decl_f32_simd512x4_11__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f32_simd512_m48_n1"]
                    unsafe fn __decl_f32_simd512x4_12__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f32_simd512_m48_n2"]
                    unsafe fn __decl_f32_simd512x4_13__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f32_simd512_m48_n3"]
                    unsafe fn __decl_f32_simd512x4_14__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f32_simd512_m48_n4"]
                    unsafe fn __decl_f32_simd512x4_15__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f32_simd512_m32_n1"]
                    unsafe fn __decl_f32_simd512x4_16__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f32_simd512_m32_n2"]
                    unsafe fn __decl_f32_simd512x4_17__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f32_simd512_m32_n3"]
                    unsafe fn __decl_f32_simd512x4_18__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f32_simd512_m32_n4"]
                    unsafe fn __decl_f32_simd512x4_19__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f32_simd512_m16_n1"]
                    unsafe fn __decl_f32_simd512x4_20__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f32_simd512_m16_n2"]
                    unsafe fn __decl_f32_simd512x4_21__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f32_simd512_m16_n3"]
                    unsafe fn __decl_f32_simd512x4_22__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f32_simd512_m16_n4"]
                    unsafe fn __decl_f32_simd512x4_23__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f32_simd512_m16_n5"]
                    unsafe fn __decl_f32_simd512x4_24__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f32_simd512_m16_n6"]
                    unsafe fn __decl_f32_simd512x4_25__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f32_simd512_m16_n7"]
                    unsafe fn __decl_f32_simd512x4_26__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f32_simd512_m16_n8"]
                    unsafe fn __decl_f32_simd512x4_27__();
                }
                pub static F32_SIMD512x4: [unsafe extern "C" fn(); 28] = [__decl_f32_simd512x4_0__,__decl_f32_simd512x4_1__,__decl_f32_simd512x4_2__,__decl_f32_simd512x4_3__,__decl_f32_simd512x4_4__,__decl_f32_simd512x4_5__,__decl_f32_simd512x4_6__,__decl_f32_simd512x4_7__,__decl_f32_simd512x4_8__,__decl_f32_simd512x4_9__,__decl_f32_simd512x4_10__,__decl_f32_simd512x4_11__,__decl_f32_simd512x4_12__,__decl_f32_simd512x4_13__,__decl_f32_simd512x4_14__,__decl_f32_simd512x4_15__,__decl_f32_simd512x4_16__,__decl_f32_simd512x4_17__,__decl_f32_simd512x4_18__,__decl_f32_simd512x4_19__,__decl_f32_simd512x4_20__,__decl_f32_simd512x4_21__,__decl_f32_simd512x4_22__,__decl_f32_simd512x4_23__,__decl_f32_simd512x4_24__,__decl_f32_simd512x4_25__,__decl_f32_simd512x4_26__,__decl_f32_simd512x4_27__,];
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c32_simd512_m48_n1"]
                    unsafe fn __decl_c32_simd512x4_0__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c32_simd512_m48_n2"]
                    unsafe fn __decl_c32_simd512x4_1__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c32_simd512_m48_n3"]
                    unsafe fn __decl_c32_simd512x4_2__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c32_simd512_m48_n4"]
                    unsafe fn __decl_c32_simd512x4_3__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c32_simd512_m40_n1"]
                    unsafe fn __decl_c32_simd512x4_4__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c32_simd512_m40_n2"]
                    unsafe fn __decl_c32_simd512x4_5__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c32_simd512_m40_n3"]
                    unsafe fn __decl_c32_simd512x4_6__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c32_simd512_m40_n4"]
                    unsafe fn __decl_c32_simd512x4_7__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c32_simd512_m32_n1"]
                    unsafe fn __decl_c32_simd512x4_8__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c32_simd512_m32_n2"]
                    unsafe fn __decl_c32_simd512x4_9__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c32_simd512_m32_n3"]
                    unsafe fn __decl_c32_simd512x4_10__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c32_simd512_m32_n4"]
                    unsafe fn __decl_c32_simd512x4_11__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c32_simd512_m24_n1"]
                    unsafe fn __decl_c32_simd512x4_12__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c32_simd512_m24_n2"]
                    unsafe fn __decl_c32_simd512x4_13__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c32_simd512_m24_n3"]
                    unsafe fn __decl_c32_simd512x4_14__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c32_simd512_m24_n4"]
                    unsafe fn __decl_c32_simd512x4_15__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c32_simd512_m16_n1"]
                    unsafe fn __decl_c32_simd512x4_16__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c32_simd512_m16_n2"]
                    unsafe fn __decl_c32_simd512x4_17__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c32_simd512_m16_n3"]
                    unsafe fn __decl_c32_simd512x4_18__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c32_simd512_m16_n4"]
                    unsafe fn __decl_c32_simd512x4_19__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c32_simd512_m8_n1"]
                    unsafe fn __decl_c32_simd512x4_20__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c32_simd512_m8_n2"]
                    unsafe fn __decl_c32_simd512x4_21__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c32_simd512_m8_n3"]
                    unsafe fn __decl_c32_simd512x4_22__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c32_simd512_m8_n4"]
                    unsafe fn __decl_c32_simd512x4_23__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c32_simd512_m8_n5"]
                    unsafe fn __decl_c32_simd512x4_24__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c32_simd512_m8_n6"]
                    unsafe fn __decl_c32_simd512x4_25__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c32_simd512_m8_n7"]
                    unsafe fn __decl_c32_simd512x4_26__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c32_simd512_m8_n8"]
                    unsafe fn __decl_c32_simd512x4_27__();
                }
                pub static C32_SIMD512x4: [unsafe extern "C" fn(); 28] = [__decl_c32_simd512x4_0__,__decl_c32_simd512x4_1__,__decl_c32_simd512x4_2__,__decl_c32_simd512x4_3__,__decl_c32_simd512x4_4__,__decl_c32_simd512x4_5__,__decl_c32_simd512x4_6__,__decl_c32_simd512x4_7__,__decl_c32_simd512x4_8__,__decl_c32_simd512x4_9__,__decl_c32_simd512x4_10__,__decl_c32_simd512x4_11__,__decl_c32_simd512x4_12__,__decl_c32_simd512x4_13__,__decl_c32_simd512x4_14__,__decl_c32_simd512x4_15__,__decl_c32_simd512x4_16__,__decl_c32_simd512x4_17__,__decl_c32_simd512x4_18__,__decl_c32_simd512x4_19__,__decl_c32_simd512x4_20__,__decl_c32_simd512x4_21__,__decl_c32_simd512x4_22__,__decl_c32_simd512x4_23__,__decl_c32_simd512x4_24__,__decl_c32_simd512x4_25__,__decl_c32_simd512x4_26__,__decl_c32_simd512x4_27__,];
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f64_simd512_m48_n1"]
                    unsafe fn __decl_f64_simd512x4_0__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f64_simd512_m48_n2"]
                    unsafe fn __decl_f64_simd512x4_1__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f64_simd512_m48_n3"]
                    unsafe fn __decl_f64_simd512x4_2__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f64_simd512_m48_n4"]
                    unsafe fn __decl_f64_simd512x4_3__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f64_simd512_m40_n1"]
                    unsafe fn __decl_f64_simd512x4_4__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f64_simd512_m40_n2"]
                    unsafe fn __decl_f64_simd512x4_5__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f64_simd512_m40_n3"]
                    unsafe fn __decl_f64_simd512x4_6__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f64_simd512_m40_n4"]
                    unsafe fn __decl_f64_simd512x4_7__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f64_simd512_m32_n1"]
                    unsafe fn __decl_f64_simd512x4_8__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f64_simd512_m32_n2"]
                    unsafe fn __decl_f64_simd512x4_9__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f64_simd512_m32_n3"]
                    unsafe fn __decl_f64_simd512x4_10__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f64_simd512_m32_n4"]
                    unsafe fn __decl_f64_simd512x4_11__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f64_simd512_m24_n1"]
                    unsafe fn __decl_f64_simd512x4_12__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f64_simd512_m24_n2"]
                    unsafe fn __decl_f64_simd512x4_13__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f64_simd512_m24_n3"]
                    unsafe fn __decl_f64_simd512x4_14__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f64_simd512_m24_n4"]
                    unsafe fn __decl_f64_simd512x4_15__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f64_simd512_m16_n1"]
                    unsafe fn __decl_f64_simd512x4_16__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f64_simd512_m16_n2"]
                    unsafe fn __decl_f64_simd512x4_17__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f64_simd512_m16_n3"]
                    unsafe fn __decl_f64_simd512x4_18__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f64_simd512_m16_n4"]
                    unsafe fn __decl_f64_simd512x4_19__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f64_simd512_m8_n1"]
                    unsafe fn __decl_f64_simd512x4_20__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f64_simd512_m8_n2"]
                    unsafe fn __decl_f64_simd512x4_21__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f64_simd512_m8_n3"]
                    unsafe fn __decl_f64_simd512x4_22__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f64_simd512_m8_n4"]
                    unsafe fn __decl_f64_simd512x4_23__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f64_simd512_m8_n5"]
                    unsafe fn __decl_f64_simd512x4_24__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f64_simd512_m8_n6"]
                    unsafe fn __decl_f64_simd512x4_25__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f64_simd512_m8_n7"]
                    unsafe fn __decl_f64_simd512x4_26__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f64_simd512_m8_n8"]
                    unsafe fn __decl_f64_simd512x4_27__();
                }
                pub static F64_SIMD512x4: [unsafe extern "C" fn(); 28] = [__decl_f64_simd512x4_0__,__decl_f64_simd512x4_1__,__decl_f64_simd512x4_2__,__decl_f64_simd512x4_3__,__decl_f64_simd512x4_4__,__decl_f64_simd512x4_5__,__decl_f64_simd512x4_6__,__decl_f64_simd512x4_7__,__decl_f64_simd512x4_8__,__decl_f64_simd512x4_9__,__decl_f64_simd512x4_10__,__decl_f64_simd512x4_11__,__decl_f64_simd512x4_12__,__decl_f64_simd512x4_13__,__decl_f64_simd512x4_14__,__decl_f64_simd512x4_15__,__decl_f64_simd512x4_16__,__decl_f64_simd512x4_17__,__decl_f64_simd512x4_18__,__decl_f64_simd512x4_19__,__decl_f64_simd512x4_20__,__decl_f64_simd512x4_21__,__decl_f64_simd512x4_22__,__decl_f64_simd512x4_23__,__decl_f64_simd512x4_24__,__decl_f64_simd512x4_25__,__decl_f64_simd512x4_26__,__decl_f64_simd512x4_27__,];
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c64_simd512_m24_n1"]
                    unsafe fn __decl_c64_simd512x4_0__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c64_simd512_m24_n2"]
                    unsafe fn __decl_c64_simd512x4_1__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c64_simd512_m24_n3"]
                    unsafe fn __decl_c64_simd512x4_2__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c64_simd512_m24_n4"]
                    unsafe fn __decl_c64_simd512x4_3__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c64_simd512_m20_n1"]
                    unsafe fn __decl_c64_simd512x4_4__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c64_simd512_m20_n2"]
                    unsafe fn __decl_c64_simd512x4_5__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c64_simd512_m20_n3"]
                    unsafe fn __decl_c64_simd512x4_6__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c64_simd512_m20_n4"]
                    unsafe fn __decl_c64_simd512x4_7__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c64_simd512_m16_n1"]
                    unsafe fn __decl_c64_simd512x4_8__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c64_simd512_m16_n2"]
                    unsafe fn __decl_c64_simd512x4_9__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c64_simd512_m16_n3"]
                    unsafe fn __decl_c64_simd512x4_10__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c64_simd512_m16_n4"]
                    unsafe fn __decl_c64_simd512x4_11__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c64_simd512_m12_n1"]
                    unsafe fn __decl_c64_simd512x4_12__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c64_simd512_m12_n2"]
                    unsafe fn __decl_c64_simd512x4_13__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c64_simd512_m12_n3"]
                    unsafe fn __decl_c64_simd512x4_14__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c64_simd512_m12_n4"]
                    unsafe fn __decl_c64_simd512x4_15__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c64_simd512_m8_n1"]
                    unsafe fn __decl_c64_simd512x4_16__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c64_simd512_m8_n2"]
                    unsafe fn __decl_c64_simd512x4_17__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c64_simd512_m8_n3"]
                    unsafe fn __decl_c64_simd512x4_18__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c64_simd512_m8_n4"]
                    unsafe fn __decl_c64_simd512x4_19__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c64_simd512_m4_n1"]
                    unsafe fn __decl_c64_simd512x4_20__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c64_simd512_m4_n2"]
                    unsafe fn __decl_c64_simd512x4_21__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c64_simd512_m4_n3"]
                    unsafe fn __decl_c64_simd512x4_22__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c64_simd512_m4_n4"]
                    unsafe fn __decl_c64_simd512x4_23__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c64_simd512_m4_n5"]
                    unsafe fn __decl_c64_simd512x4_24__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c64_simd512_m4_n6"]
                    unsafe fn __decl_c64_simd512x4_25__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c64_simd512_m4_n7"]
                    unsafe fn __decl_c64_simd512x4_26__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c64_simd512_m4_n8"]
                    unsafe fn __decl_c64_simd512x4_27__();
                }
                pub static C64_SIMD512x4: [unsafe extern "C" fn(); 28] = [__decl_c64_simd512x4_0__,__decl_c64_simd512x4_1__,__decl_c64_simd512x4_2__,__decl_c64_simd512x4_3__,__decl_c64_simd512x4_4__,__decl_c64_simd512x4_5__,__decl_c64_simd512x4_6__,__decl_c64_simd512x4_7__,__decl_c64_simd512x4_8__,__decl_c64_simd512x4_9__,__decl_c64_simd512x4_10__,__decl_c64_simd512x4_11__,__decl_c64_simd512x4_12__,__decl_c64_simd512x4_13__,__decl_c64_simd512x4_14__,__decl_c64_simd512x4_15__,__decl_c64_simd512x4_16__,__decl_c64_simd512x4_17__,__decl_c64_simd512x4_18__,__decl_c64_simd512x4_19__,__decl_c64_simd512x4_20__,__decl_c64_simd512x4_21__,__decl_c64_simd512x4_22__,__decl_c64_simd512x4_23__,__decl_c64_simd512x4_24__,__decl_c64_simd512x4_25__,__decl_c64_simd512x4_26__,__decl_c64_simd512x4_27__,];
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f32_simd256_m24_n1"]
                    unsafe fn __decl_f32_simd256_0__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f32_simd256_m24_n2"]
                    unsafe fn __decl_f32_simd256_1__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f32_simd256_m24_n3"]
                    unsafe fn __decl_f32_simd256_2__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f32_simd256_m24_n4"]
                    unsafe fn __decl_f32_simd256_3__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f32_simd256_m16_n1"]
                    unsafe fn __decl_f32_simd256_4__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f32_simd256_m16_n2"]
                    unsafe fn __decl_f32_simd256_5__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f32_simd256_m16_n3"]
                    unsafe fn __decl_f32_simd256_6__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f32_simd256_m16_n4"]
                    unsafe fn __decl_f32_simd256_7__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f32_simd256_m8_n1"]
                    unsafe fn __decl_f32_simd256_8__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f32_simd256_m8_n2"]
                    unsafe fn __decl_f32_simd256_9__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f32_simd256_m8_n3"]
                    unsafe fn __decl_f32_simd256_10__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f32_simd256_m8_n4"]
                    unsafe fn __decl_f32_simd256_11__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f32_simd256_m8_n5"]
                    unsafe fn __decl_f32_simd256_12__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f32_simd256_m8_n6"]
                    unsafe fn __decl_f32_simd256_13__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f32_simd256_m8_n7"]
                    unsafe fn __decl_f32_simd256_14__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f32_simd256_m8_n8"]
                    unsafe fn __decl_f32_simd256_15__();
                }
                pub static F32_SIMD256: [unsafe extern "C" fn(); 16] = [__decl_f32_simd256_0__,__decl_f32_simd256_1__,__decl_f32_simd256_2__,__decl_f32_simd256_3__,__decl_f32_simd256_4__,__decl_f32_simd256_5__,__decl_f32_simd256_6__,__decl_f32_simd256_7__,__decl_f32_simd256_8__,__decl_f32_simd256_9__,__decl_f32_simd256_10__,__decl_f32_simd256_11__,__decl_f32_simd256_12__,__decl_f32_simd256_13__,__decl_f32_simd256_14__,__decl_f32_simd256_15__,];
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c32_simd256_m12_n1"]
                    unsafe fn __decl_c32_simd256_0__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c32_simd256_m12_n2"]
                    unsafe fn __decl_c32_simd256_1__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c32_simd256_m12_n3"]
                    unsafe fn __decl_c32_simd256_2__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c32_simd256_m12_n4"]
                    unsafe fn __decl_c32_simd256_3__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c32_simd256_m8_n1"]
                    unsafe fn __decl_c32_simd256_4__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c32_simd256_m8_n2"]
                    unsafe fn __decl_c32_simd256_5__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c32_simd256_m8_n3"]
                    unsafe fn __decl_c32_simd256_6__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c32_simd256_m8_n4"]
                    unsafe fn __decl_c32_simd256_7__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c32_simd256_m4_n1"]
                    unsafe fn __decl_c32_simd256_8__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c32_simd256_m4_n2"]
                    unsafe fn __decl_c32_simd256_9__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c32_simd256_m4_n3"]
                    unsafe fn __decl_c32_simd256_10__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c32_simd256_m4_n4"]
                    unsafe fn __decl_c32_simd256_11__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c32_simd256_m4_n5"]
                    unsafe fn __decl_c32_simd256_12__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c32_simd256_m4_n6"]
                    unsafe fn __decl_c32_simd256_13__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c32_simd256_m4_n7"]
                    unsafe fn __decl_c32_simd256_14__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c32_simd256_m4_n8"]
                    unsafe fn __decl_c32_simd256_15__();
                }
                pub static C32_SIMD256: [unsafe extern "C" fn(); 16] = [__decl_c32_simd256_0__,__decl_c32_simd256_1__,__decl_c32_simd256_2__,__decl_c32_simd256_3__,__decl_c32_simd256_4__,__decl_c32_simd256_5__,__decl_c32_simd256_6__,__decl_c32_simd256_7__,__decl_c32_simd256_8__,__decl_c32_simd256_9__,__decl_c32_simd256_10__,__decl_c32_simd256_11__,__decl_c32_simd256_12__,__decl_c32_simd256_13__,__decl_c32_simd256_14__,__decl_c32_simd256_15__,];
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f64_simd256_m12_n1"]
                    unsafe fn __decl_f64_simd256_0__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f64_simd256_m12_n2"]
                    unsafe fn __decl_f64_simd256_1__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f64_simd256_m12_n3"]
                    unsafe fn __decl_f64_simd256_2__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f64_simd256_m12_n4"]
                    unsafe fn __decl_f64_simd256_3__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f64_simd256_m8_n1"]
                    unsafe fn __decl_f64_simd256_4__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f64_simd256_m8_n2"]
                    unsafe fn __decl_f64_simd256_5__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f64_simd256_m8_n3"]
                    unsafe fn __decl_f64_simd256_6__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f64_simd256_m8_n4"]
                    unsafe fn __decl_f64_simd256_7__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f64_simd256_m4_n1"]
                    unsafe fn __decl_f64_simd256_8__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f64_simd256_m4_n2"]
                    unsafe fn __decl_f64_simd256_9__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f64_simd256_m4_n3"]
                    unsafe fn __decl_f64_simd256_10__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f64_simd256_m4_n4"]
                    unsafe fn __decl_f64_simd256_11__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f64_simd256_m4_n5"]
                    unsafe fn __decl_f64_simd256_12__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f64_simd256_m4_n6"]
                    unsafe fn __decl_f64_simd256_13__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f64_simd256_m4_n7"]
                    unsafe fn __decl_f64_simd256_14__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f64_simd256_m4_n8"]
                    unsafe fn __decl_f64_simd256_15__();
                }
                pub static F64_SIMD256: [unsafe extern "C" fn(); 16] = [__decl_f64_simd256_0__,__decl_f64_simd256_1__,__decl_f64_simd256_2__,__decl_f64_simd256_3__,__decl_f64_simd256_4__,__decl_f64_simd256_5__,__decl_f64_simd256_6__,__decl_f64_simd256_7__,__decl_f64_simd256_8__,__decl_f64_simd256_9__,__decl_f64_simd256_10__,__decl_f64_simd256_11__,__decl_f64_simd256_12__,__decl_f64_simd256_13__,__decl_f64_simd256_14__,__decl_f64_simd256_15__,];
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c64_simd256_m6_n1"]
                    unsafe fn __decl_c64_simd256_0__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c64_simd256_m6_n2"]
                    unsafe fn __decl_c64_simd256_1__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c64_simd256_m6_n3"]
                    unsafe fn __decl_c64_simd256_2__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c64_simd256_m6_n4"]
                    unsafe fn __decl_c64_simd256_3__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c64_simd256_m4_n1"]
                    unsafe fn __decl_c64_simd256_4__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c64_simd256_m4_n2"]
                    unsafe fn __decl_c64_simd256_5__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c64_simd256_m4_n3"]
                    unsafe fn __decl_c64_simd256_6__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c64_simd256_m4_n4"]
                    unsafe fn __decl_c64_simd256_7__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c64_simd256_m2_n1"]
                    unsafe fn __decl_c64_simd256_8__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c64_simd256_m2_n2"]
                    unsafe fn __decl_c64_simd256_9__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c64_simd256_m2_n3"]
                    unsafe fn __decl_c64_simd256_10__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c64_simd256_m2_n4"]
                    unsafe fn __decl_c64_simd256_11__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c64_simd256_m2_n5"]
                    unsafe fn __decl_c64_simd256_12__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c64_simd256_m2_n6"]
                    unsafe fn __decl_c64_simd256_13__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c64_simd256_m2_n7"]
                    unsafe fn __decl_c64_simd256_14__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c64_simd256_m2_n8"]
                    unsafe fn __decl_c64_simd256_15__();
                }
                pub static C64_SIMD256: [unsafe extern "C" fn(); 16] = [__decl_c64_simd256_0__,__decl_c64_simd256_1__,__decl_c64_simd256_2__,__decl_c64_simd256_3__,__decl_c64_simd256_4__,__decl_c64_simd256_5__,__decl_c64_simd256_6__,__decl_c64_simd256_7__,__decl_c64_simd256_8__,__decl_c64_simd256_9__,__decl_c64_simd256_10__,__decl_c64_simd256_11__,__decl_c64_simd256_12__,__decl_c64_simd256_13__,__decl_c64_simd256_14__,__decl_c64_simd256_15__,];
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f32_simd128_m4_n1"]
                    unsafe fn __decl_f32_simd128_0__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f32_simd128_m4_n2"]
                    unsafe fn __decl_f32_simd128_1__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f32_simd128_m4_n3"]
                    unsafe fn __decl_f32_simd128_2__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f32_simd128_m4_n4"]
                    unsafe fn __decl_f32_simd128_3__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f32_simd128_m4_n5"]
                    unsafe fn __decl_f32_simd128_4__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f32_simd128_m4_n6"]
                    unsafe fn __decl_f32_simd128_5__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f32_simd128_m4_n7"]
                    unsafe fn __decl_f32_simd128_6__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f32_simd128_m4_n8"]
                    unsafe fn __decl_f32_simd128_7__();
                }
                pub static F32_SIMD128: [unsafe extern "C" fn(); 8] = [__decl_f32_simd128_0__,__decl_f32_simd128_1__,__decl_f32_simd128_2__,__decl_f32_simd128_3__,__decl_f32_simd128_4__,__decl_f32_simd128_5__,__decl_f32_simd128_6__,__decl_f32_simd128_7__,];
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c32_simd128_m2_n1"]
                    unsafe fn __decl_c32_simd128_0__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c32_simd128_m2_n2"]
                    unsafe fn __decl_c32_simd128_1__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c32_simd128_m2_n3"]
                    unsafe fn __decl_c32_simd128_2__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c32_simd128_m2_n4"]
                    unsafe fn __decl_c32_simd128_3__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c32_simd128_m2_n5"]
                    unsafe fn __decl_c32_simd128_4__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c32_simd128_m2_n6"]
                    unsafe fn __decl_c32_simd128_5__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c32_simd128_m2_n7"]
                    unsafe fn __decl_c32_simd128_6__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c32_simd128_m2_n8"]
                    unsafe fn __decl_c32_simd128_7__();
                }
                pub static C32_SIMD128: [unsafe extern "C" fn(); 8] = [__decl_c32_simd128_0__,__decl_c32_simd128_1__,__decl_c32_simd128_2__,__decl_c32_simd128_3__,__decl_c32_simd128_4__,__decl_c32_simd128_5__,__decl_c32_simd128_6__,__decl_c32_simd128_7__,];
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f64_simd128_m2_n1"]
                    unsafe fn __decl_f64_simd128_0__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f64_simd128_m2_n2"]
                    unsafe fn __decl_f64_simd128_1__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f64_simd128_m2_n3"]
                    unsafe fn __decl_f64_simd128_2__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f64_simd128_m2_n4"]
                    unsafe fn __decl_f64_simd128_3__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f64_simd128_m2_n5"]
                    unsafe fn __decl_f64_simd128_4__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f64_simd128_m2_n6"]
                    unsafe fn __decl_f64_simd128_5__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f64_simd128_m2_n7"]
                    unsafe fn __decl_f64_simd128_6__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f64_simd128_m2_n8"]
                    unsafe fn __decl_f64_simd128_7__();
                }
                pub static F64_SIMD128: [unsafe extern "C" fn(); 8] = [__decl_f64_simd128_0__,__decl_f64_simd128_1__,__decl_f64_simd128_2__,__decl_f64_simd128_3__,__decl_f64_simd128_4__,__decl_f64_simd128_5__,__decl_f64_simd128_6__,__decl_f64_simd128_7__,];
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c64_simd128_m1_n1"]
                    unsafe fn __decl_c64_simd128_0__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c64_simd128_m1_n2"]
                    unsafe fn __decl_c64_simd128_1__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c64_simd128_m1_n3"]
                    unsafe fn __decl_c64_simd128_2__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c64_simd128_m1_n4"]
                    unsafe fn __decl_c64_simd128_3__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c64_simd128_m1_n5"]
                    unsafe fn __decl_c64_simd128_4__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c64_simd128_m1_n6"]
                    unsafe fn __decl_c64_simd128_5__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c64_simd128_m1_n7"]
                    unsafe fn __decl_c64_simd128_6__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c64_simd128_m1_n8"]
                    unsafe fn __decl_c64_simd128_7__();
                }
                pub static C64_SIMD128: [unsafe extern "C" fn(); 8] = [__decl_c64_simd128_0__,__decl_c64_simd128_1__,__decl_c64_simd128_2__,__decl_c64_simd128_3__,__decl_c64_simd128_4__,__decl_c64_simd128_5__,__decl_c64_simd128_6__,__decl_c64_simd128_7__,];
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f32_simd64_m2_n1"]
                    unsafe fn __decl_f32_simd64_0__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f32_simd64_m2_n2"]
                    unsafe fn __decl_f32_simd64_1__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f32_simd64_m2_n3"]
                    unsafe fn __decl_f32_simd64_2__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f32_simd64_m2_n4"]
                    unsafe fn __decl_f32_simd64_3__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f32_simd64_m2_n5"]
                    unsafe fn __decl_f32_simd64_4__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f32_simd64_m2_n6"]
                    unsafe fn __decl_f32_simd64_5__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f32_simd64_m2_n7"]
                    unsafe fn __decl_f32_simd64_6__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f32_simd64_m2_n8"]
                    unsafe fn __decl_f32_simd64_7__();
                }
                pub static F32_SIMD64: [unsafe extern "C" fn(); 8] = [__decl_f32_simd64_0__,__decl_f32_simd64_1__,__decl_f32_simd64_2__,__decl_f32_simd64_3__,__decl_f32_simd64_4__,__decl_f32_simd64_5__,__decl_f32_simd64_6__,__decl_f32_simd64_7__,];
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c32_simd64_m1_n1"]
                    unsafe fn __decl_c32_simd64_0__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c32_simd64_m1_n2"]
                    unsafe fn __decl_c32_simd64_1__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c32_simd64_m1_n3"]
                    unsafe fn __decl_c32_simd64_2__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c32_simd64_m1_n4"]
                    unsafe fn __decl_c32_simd64_3__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c32_simd64_m1_n5"]
                    unsafe fn __decl_c32_simd64_4__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c32_simd64_m1_n6"]
                    unsafe fn __decl_c32_simd64_5__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c32_simd64_m1_n7"]
                    unsafe fn __decl_c32_simd64_6__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_c32_simd64_m1_n8"]
                    unsafe fn __decl_c32_simd64_7__();
                }
                pub static C32_SIMD64: [unsafe extern "C" fn(); 8] = [__decl_c32_simd64_0__,__decl_c32_simd64_1__,__decl_c32_simd64_2__,__decl_c32_simd64_3__,__decl_c32_simd64_4__,__decl_c32_simd64_5__,__decl_c32_simd64_6__,__decl_c32_simd64_7__,];
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f64_simd64_m1_n1"]
                    unsafe fn __decl_f64_simd64_0__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f64_simd64_m1_n2"]
                    unsafe fn __decl_f64_simd64_1__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f64_simd64_m1_n3"]
                    unsafe fn __decl_f64_simd64_2__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f64_simd64_m1_n4"]
                    unsafe fn __decl_f64_simd64_3__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f64_simd64_m1_n5"]
                    unsafe fn __decl_f64_simd64_4__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f64_simd64_m1_n6"]
                    unsafe fn __decl_f64_simd64_5__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f64_simd64_m1_n7"]
                    unsafe fn __decl_f64_simd64_6__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_microkernel_f64_simd64_m1_n8"]
                    unsafe fn __decl_f64_simd64_7__();
                }
                pub static F64_SIMD64: [unsafe extern "C" fn(); 8] = [__decl_f64_simd64_0__,__decl_f64_simd64_1__,__decl_f64_simd64_2__,__decl_f64_simd64_3__,__decl_f64_simd64_4__,__decl_f64_simd64_5__,__decl_f64_simd64_6__,__decl_f64_simd64_7__,];
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_pack_f32_simd512_m96"]
                    unsafe fn __decl_f32_simdpack_512_0__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_pack_f32_simd512_m80"]
                    unsafe fn __decl_f32_simdpack_512_1__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_pack_f32_simd512_m64"]
                    unsafe fn __decl_f32_simdpack_512_2__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_pack_f32_simd512_m48"]
                    unsafe fn __decl_f32_simdpack_512_3__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_pack_f32_simd512_m32"]
                    unsafe fn __decl_f32_simdpack_512_4__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_pack_f32_simd512_m16"]
                    unsafe fn __decl_f32_simdpack_512_5__();
                }
                pub static F32_SIMDpack_512: [unsafe extern "C" fn(); 6] = [__decl_f32_simdpack_512_0__,__decl_f32_simdpack_512_1__,__decl_f32_simdpack_512_2__,__decl_f32_simdpack_512_3__,__decl_f32_simdpack_512_4__,__decl_f32_simdpack_512_5__,];
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_pack_f64_simd512_m48"]
                    unsafe fn __decl_f64_simdpack_512_0__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_pack_f64_simd512_m40"]
                    unsafe fn __decl_f64_simdpack_512_1__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_pack_f64_simd512_m32"]
                    unsafe fn __decl_f64_simdpack_512_2__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_pack_f64_simd512_m24"]
                    unsafe fn __decl_f64_simdpack_512_3__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_pack_f64_simd512_m16"]
                    unsafe fn __decl_f64_simdpack_512_4__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_pack_f64_simd512_m8"]
                    unsafe fn __decl_f64_simdpack_512_5__();
                }
                pub static F64_SIMDpack_512: [unsafe extern "C" fn(); 6] = [__decl_f64_simdpack_512_0__,__decl_f64_simdpack_512_1__,__decl_f64_simdpack_512_2__,__decl_f64_simdpack_512_3__,__decl_f64_simdpack_512_4__,__decl_f64_simdpack_512_5__,];
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_pack_c32_simd512_m48"]
                    unsafe fn __decl_c32_simdpack_512_0__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_pack_c32_simd512_m40"]
                    unsafe fn __decl_c32_simdpack_512_1__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_pack_c32_simd512_m32"]
                    unsafe fn __decl_c32_simdpack_512_2__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_pack_c32_simd512_m24"]
                    unsafe fn __decl_c32_simdpack_512_3__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_pack_c32_simd512_m16"]
                    unsafe fn __decl_c32_simdpack_512_4__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_pack_c32_simd512_m8"]
                    unsafe fn __decl_c32_simdpack_512_5__();
                }
                pub static C32_SIMDpack_512: [unsafe extern "C" fn(); 6] = [__decl_c32_simdpack_512_0__,__decl_c32_simdpack_512_1__,__decl_c32_simdpack_512_2__,__decl_c32_simdpack_512_3__,__decl_c32_simdpack_512_4__,__decl_c32_simdpack_512_5__,];
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_pack_c64_simd512_m24"]
                    unsafe fn __decl_c64_simdpack_512_0__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_pack_c64_simd512_m20"]
                    unsafe fn __decl_c64_simdpack_512_1__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_pack_c64_simd512_m16"]
                    unsafe fn __decl_c64_simdpack_512_2__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_pack_c64_simd512_m12"]
                    unsafe fn __decl_c64_simdpack_512_3__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_pack_c64_simd512_m8"]
                    unsafe fn __decl_c64_simdpack_512_4__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_pack_c64_simd512_m4"]
                    unsafe fn __decl_c64_simdpack_512_5__();
                }
                pub static C64_SIMDpack_512: [unsafe extern "C" fn(); 6] = [__decl_c64_simdpack_512_0__,__decl_c64_simdpack_512_1__,__decl_c64_simdpack_512_2__,__decl_c64_simdpack_512_3__,__decl_c64_simdpack_512_4__,__decl_c64_simdpack_512_5__,];
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_pack_f32_simd256_m24"]
                    unsafe fn __decl_f32_simdpack_256_0__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_pack_f32_simd256_m16"]
                    unsafe fn __decl_f32_simdpack_256_1__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_pack_f32_simd256_m8"]
                    unsafe fn __decl_f32_simdpack_256_2__();
                }
                pub static F32_SIMDpack_256: [unsafe extern "C" fn(); 3] = [__decl_f32_simdpack_256_0__,__decl_f32_simdpack_256_1__,__decl_f32_simdpack_256_2__,];
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_pack_f64_simd256_m12"]
                    unsafe fn __decl_f64_simdpack_256_0__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_pack_f64_simd256_m8"]
                    unsafe fn __decl_f64_simdpack_256_1__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_pack_f64_simd256_m4"]
                    unsafe fn __decl_f64_simdpack_256_2__();
                }
                pub static F64_SIMDpack_256: [unsafe extern "C" fn(); 3] = [__decl_f64_simdpack_256_0__,__decl_f64_simdpack_256_1__,__decl_f64_simdpack_256_2__,];
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_pack_c32_simd256_m12"]
                    unsafe fn __decl_c32_simdpack_256_0__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_pack_c32_simd256_m8"]
                    unsafe fn __decl_c32_simdpack_256_1__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_pack_c32_simd256_m4"]
                    unsafe fn __decl_c32_simdpack_256_2__();
                }
                pub static C32_SIMDpack_256: [unsafe extern "C" fn(); 3] = [__decl_c32_simdpack_256_0__,__decl_c32_simdpack_256_1__,__decl_c32_simdpack_256_2__,];
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_pack_c64_simd256_m6"]
                    unsafe fn __decl_c64_simdpack_256_0__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_pack_c64_simd256_m4"]
                    unsafe fn __decl_c64_simdpack_256_1__();
                }
                
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_pack_c64_simd256_m2"]
                    unsafe fn __decl_c64_simdpack_256_2__();
                }
                pub static C64_SIMDpack_256: [unsafe extern "C" fn(); 3] = [__decl_c64_simdpack_256_0__,__decl_c64_simdpack_256_1__,__decl_c64_simdpack_256_2__,];
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_pack_f32_simd128_m4"]
                    unsafe fn __decl_f32_simdpack_128_0__();
                }
                pub static F32_SIMDpack_128: [unsafe extern "C" fn(); 1] = [__decl_f32_simdpack_128_0__,];
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_pack_f64_simd128_m2"]
                    unsafe fn __decl_f64_simdpack_128_0__();
                }
                pub static F64_SIMDpack_128: [unsafe extern "C" fn(); 1] = [__decl_f64_simdpack_128_0__,];
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_pack_c32_simd128_m2"]
                    unsafe fn __decl_c32_simdpack_128_0__();
                }
                pub static C32_SIMDpack_128: [unsafe extern "C" fn(); 1] = [__decl_c32_simdpack_128_0__,];
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_pack_c64_simd128_m1"]
                    unsafe fn __decl_c64_simdpack_128_0__();
                }
                pub static C64_SIMDpack_128: [unsafe extern "C" fn(); 1] = [__decl_c64_simdpack_128_0__,];
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_pack_f32_simd64_m2"]
                    unsafe fn __decl_f32_simdpack_64_0__();
                }
                pub static F32_SIMDpack_64: [unsafe extern "C" fn(); 1] = [__decl_f32_simdpack_64_0__,];
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_pack_c32_simd64_m1"]
                    unsafe fn __decl_c32_simdpack_64_0__();
                }
                pub static C32_SIMDpack_64: [unsafe extern "C" fn(); 1] = [__decl_c32_simdpack_64_0__,];
                unsafe extern "C" {
                    #[link_name = "gemm_v0_gemm_pack_f64_simd64_m1"]
                    unsafe fn __decl_f64_simdpack_64_0__();
                }
                pub static F64_SIMDpack_64: [unsafe extern "C" fn(); 1] = [__decl_f64_simdpack_64_0__,];