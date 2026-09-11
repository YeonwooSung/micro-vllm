// Vendored Metal kernels extracted from colibri c/backend_metal.mm
// License: Apache-2.0 (see src/gpu/vendor/NOTICE)
// Do not treat this as our original implementation.


#include <metal_stdlib>
using namespace metal;

// fmt=6 E8/IQ3 magnitude grid — generated from quant.h e8_grid (must stay identical;
// the metal-test oracle compares against the CPU decoder, so drift fails the build).
constant uchar4 E8G[256] = {
  uchar4(4,4,4,4),uchar4(20,4,4,4),uchar4(36,4,4,4),uchar4(12,12,4,4),uchar4(28,12,4,4),uchar4(62,12,4,4),uchar4(4,20,4,4),uchar4(20,20,4,4),
  uchar4(12,28,4,4),uchar4(20,36,4,4),uchar4(28,62,4,4),uchar4(44,62,4,4),uchar4(12,4,12,4),uchar4(28,4,12,4),uchar4(4,12,12,4),uchar4(20,12,12,4),
  uchar4(12,20,12,4),uchar4(44,20,12,4),uchar4(4,28,12,4),uchar4(20,28,12,4),uchar4(12,36,12,4),uchar4(36,44,12,4),uchar4(4,62,12,4),uchar4(4,4,20,4),
  uchar4(20,4,20,4),uchar4(36,4,20,4),uchar4(12,12,20,4),uchar4(4,20,20,4),uchar4(20,20,20,4),uchar4(12,28,20,4),uchar4(28,28,20,4),uchar4(62,28,20,4),
  uchar4(12,44,20,4),uchar4(62,44,20,4),uchar4(44,62,20,4),uchar4(12,4,28,4),uchar4(62,4,28,4),uchar4(4,12,28,4),uchar4(20,12,28,4),uchar4(44,20,28,4),
  uchar4(4,62,28,4),uchar4(28,12,36,4),uchar4(62,28,36,4),uchar4(36,36,36,4),uchar4(62,44,36,4),uchar4(28,62,36,4),uchar4(44,62,36,4),uchar4(12,4,44,4),
  uchar4(62,4,44,4),uchar4(20,28,44,4),uchar4(20,44,44,4),uchar4(44,28,52,4),uchar4(36,52,52,4),uchar4(4,12,62,4),uchar4(36,12,62,4),uchar4(52,12,62,4),
  uchar4(28,36,62,4),uchar4(12,52,62,4),uchar4(12,4,4,12),uchar4(28,4,4,12),uchar4(4,12,4,12),uchar4(20,12,4,12),uchar4(12,20,4,12),uchar4(28,20,4,12),
  uchar4(4,28,4,12),uchar4(20,28,4,12),uchar4(36,28,4,12),uchar4(62,36,4,12),uchar4(4,44,4,12),uchar4(4,4,12,12),uchar4(20,4,12,12),uchar4(12,12,12,12),
  uchar4(4,20,12,12),uchar4(20,20,12,12),uchar4(12,4,20,12),uchar4(28,4,20,12),uchar4(4,12,20,12),uchar4(20,12,20,12),uchar4(12,20,20,12),uchar4(4,28,20,12),
  uchar4(20,62,20,12),uchar4(4,4,28,12),uchar4(20,4,28,12),uchar4(4,20,28,12),uchar4(12,28,28,12),uchar4(52,36,28,12),uchar4(52,52,28,12),uchar4(12,4,36,12),
  uchar4(44,4,36,12),uchar4(4,44,36,12),uchar4(4,20,44,12),uchar4(36,20,44,12),uchar4(52,36,44,12),uchar4(12,62,44,12),uchar4(44,4,52,12),uchar4(20,20,62,12),
  uchar4(4,36,62,12),uchar4(4,4,4,20),uchar4(20,4,4,20),uchar4(12,12,4,20),uchar4(28,12,4,20),uchar4(4,20,4,20),uchar4(20,20,4,20),uchar4(52,20,4,20),
  uchar4(12,28,4,20),uchar4(20,36,4,20),uchar4(12,4,12,20),uchar4(28,4,12,20),uchar4(44,4,12,20),uchar4(4,12,12,20),uchar4(20,12,12,20),uchar4(12,20,12,20),
  uchar4(4,28,12,20),uchar4(28,52,12,20),uchar4(62,52,12,20),uchar4(4,62,12,20),uchar4(4,4,20,20),uchar4(20,4,20,20),uchar4(12,12,20,20),uchar4(62,12,20,20),
  uchar4(4,20,20,20),uchar4(20,20,20,20),uchar4(62,28,20,20),uchar4(4,36,20,20),uchar4(44,44,20,20),uchar4(12,4,28,20),uchar4(4,12,28,20),uchar4(36,12,28,20),
  uchar4(4,62,28,20),uchar4(36,62,28,20),uchar4(44,28,36,20),uchar4(28,44,36,20),uchar4(28,4,44,20),uchar4(62,20,44,20),uchar4(12,36,44,20),uchar4(36,62,44,20),
  uchar4(12,4,62,20),uchar4(28,4,62,20),uchar4(52,12,62,20),uchar4(44,36,62,20),uchar4(12,4,4,28),uchar4(4,12,4,28),uchar4(20,12,4,28),uchar4(12,20,4,28),
  uchar4(28,20,4,28),uchar4(4,44,4,28),uchar4(44,52,4,28),uchar4(20,62,4,28),uchar4(4,4,12,28),uchar4(20,4,12,28),uchar4(4,20,12,28),uchar4(12,28,12,28),
  uchar4(36,36,12,28),uchar4(52,36,12,28),uchar4(12,4,20,28),uchar4(28,4,20,28),uchar4(4,12,20,28),uchar4(44,20,20,28),uchar4(20,44,20,28),uchar4(20,62,20,28),
  uchar4(12,12,28,28),uchar4(28,28,28,28),uchar4(4,28,36,28),uchar4(62,36,36,28),uchar4(20,62,36,28),uchar4(4,4,44,28),uchar4(52,4,44,28),uchar4(20,20,44,28),
  uchar4(44,44,44,28),uchar4(36,12,52,28),uchar4(52,28,52,28),uchar4(28,52,52,28),uchar4(28,28,62,28),uchar4(4,52,62,28),uchar4(36,4,4,36),uchar4(62,12,4,36),
  uchar4(44,28,4,36),uchar4(62,28,4,36),uchar4(28,44,4,36),uchar4(62,44,4,36),uchar4(36,62,12,36),uchar4(4,20,20,36),uchar4(62,28,20,36),uchar4(4,36,20,36),
  uchar4(4,52,20,36),uchar4(52,52,20,36),uchar4(62,4,28,36),uchar4(44,36,28,36),uchar4(36,4,36,36),uchar4(12,44,36,36),uchar4(36,52,36,36),uchar4(44,20,44,36),
  uchar4(28,36,44,36),uchar4(4,62,44,36),uchar4(44,4,62,36),uchar4(4,12,62,36),uchar4(20,12,62,36),uchar4(4,28,62,36),uchar4(20,12,4,44),uchar4(12,36,4,44),
  uchar4(4,62,4,44),uchar4(4,4,12,44),uchar4(52,4,12,44),uchar4(52,20,12,44),uchar4(44,44,12,44),uchar4(36,12,20,44),uchar4(20,28,20,44),uchar4(20,62,20,44),
  uchar4(20,4,28,44),uchar4(28,44,28,44),uchar4(4,12,36,44),uchar4(28,20,36,44),uchar4(62,20,36,44),uchar4(20,62,36,44),uchar4(20,4,44,44),uchar4(12,28,44,44),
  uchar4(4,44,52,44),uchar4(36,20,62,44),uchar4(20,36,62,44),uchar4(36,20,4,52),uchar4(36,36,4,52),uchar4(52,36,4,52),uchar4(36,52,4,52),uchar4(12,20,12,52),
  uchar4(12,52,12,52),uchar4(62,12,20,52),uchar4(36,52,20,52),uchar4(4,28,28,52),uchar4(52,28,28,52),uchar4(36,36,36,52),uchar4(44,4,44,52),uchar4(20,44,44,52),
  uchar4(28,28,52,52),uchar4(28,4,62,52),uchar4(12,20,62,52),uchar4(28,4,4,62),uchar4(44,4,4,62),uchar4(62,4,4,62),uchar4(4,12,4,62),uchar4(20,28,4,62),
  uchar4(20,44,4,62),uchar4(52,20,12,62),uchar4(4,36,12,62),uchar4(20,12,20,62),uchar4(44,36,20,62),uchar4(20,44,20,62),uchar4(4,4,28,62),uchar4(44,12,28,62),
  uchar4(28,28,28,62),uchar4(4,52,28,62),uchar4(12,20,36,62),uchar4(12,36,36,62),uchar4(4,4,44,62),uchar4(20,4,44,62),uchar4(36,20,44,62),uchar4(4,28,52,62)
};

kernel void mm_gemv(device const uchar* w      [[buffer(0)]],   // raw weight bytes
                    device const float* scale  [[buffer(1)]],   // [O] (fmt<4) or [O,ceil(I/gsz)] (fmt==4)
                    device const float* x      [[buffer(2)]],   // [S,I]
                    device float*       y      [[buffer(3)]],   // [S,O]
                    constant int& S [[buffer(4)]], constant int& I [[buffer(5)]],
                    constant int& O [[buffer(6)]], constant int& fmt [[buffer(7)]],
                    constant int& NT [[buffer(8)]],
                    constant int& gsz [[buffer(9)]],             // fmt==4 group size (ignored otherwise)
                    uint tg [[threadgroup_position_in_grid]],
                    uint slane [[thread_index_in_simdgroup]],
                    uint sgid [[simdgroup_index_in_threadgroup]]) {
  // one SIMDGROUP per output element, 4 per threadgroup, 8-value loads (see moe_gemv)
  long row = (long)tg*4 + sgid; if (row >= NT) return;
  int o = row % O, si = row / O;
  device const float* xr = x + (long)si * I;
  device const float4* x4 = (device const float4*)xr;
  int I8 = (I & 7) ? 0 : (I/8);
  float acc = 0.0f;
  if (fmt == 1) {                                   // int8
    device const char* wr = (device const char*)(w) + (long)o * I;
    device const char4* w4 = (device const char4*)wr;
    for (int c = slane; c < I8; c += 32) acc += dot(float4(w4[2*c]),x4[2*c]) + dot(float4(w4[2*c+1]),x4[2*c+1]);
    for (int i = I8*8 + slane; i < I; i += 32) acc += float(wr[i]) * xr[i];
  } else if (fmt == 2) {                            // int4 packed, rb=(I+1)/2
    int rb = (I+1)/2;
    device const uchar* wr = w + (long)o * rb;
    device const uchar4* w4 = (device const uchar4*)wr;
    for (int c = slane; c < I8; c += 32) { uchar4 b = w4[c];
      float4 w0 = float4(float(int(b.x&0xF)-8), float(int(b.x>>4)-8), float(int(b.y&0xF)-8), float(int(b.y>>4)-8));
      float4 w1 = float4(float(int(b.z&0xF)-8), float(int(b.z>>4)-8), float(int(b.w&0xF)-8), float(int(b.w>>4)-8));
      acc += dot(w0,x4[2*c]) + dot(w1,x4[2*c+1]);
    }
    for (int i = I8*8 + slane; i < I; i += 32) {
      uchar b = wr[i>>1]; int v = (i&1) ? (b>>4) : (b&0xF); acc += float(v-8) * xr[i];
    }
  } else if (fmt == 3) {                            // int2 packed, rb=(I+3)/4
    int rb = (I+3)/4;
    device const uchar* wr = w + (long)o * rb;
    for (int i = slane; i < I; i += 32) {
      uchar b = wr[i>>2]; int v = (b >> (2*(i&3))) & 0x3; acc += float(v-2) * xr[i];
    }
  } else if (fmt == 4) {                            // int4 GROUPED: same nibble packing as fmt=2,
                                                     // one f32 scale per gsz-element group along I.
                                                     // Each lane owns one packed byte (2 elements) per
                                                     // 64-lane stride -> memory access stays coalesced,
                                                     // and a group never splits a byte (gsz is even).
    int rb = (I+1)/2; int ng = (I+gsz-1)/gsz;
    device const uchar* wr = w + (long)o * rb;
    device const float* scl = scale + (long)o * ng;
    for (int i = slane*2; i < I; i += 64) {
      uchar b = wr[i>>1];
      int g0 = i / gsz; float sc0 = scl[g0];
      acc += float(int(b&0xF)-8) * xr[i] * sc0;
      if (i+1 < I) {
        int g1 = (i+1) / gsz; float sc1 = (g1==g0) ? sc0 : scl[g1];
        acc += float(int(b>>4)-8) * xr[i+1] * sc1;
      }
    }
  } else if (fmt == 8) {                            // fp8 e4m3 passthrough: one raw byte per
                                                     // element (like fmt=1), scale per 128x128
                                                     // block folded into acc (like a grouped fmt).
    int nblkI = (I + 127) / 128;
    device const uchar* wr = w + (long)o * I;
    device const float* scl = scale + (long)(o/128) * nblkI;
    for (int i = slane; i < I; i += 32) {
      uchar b = wr[i];
      uint sign = b >> 7, exp = (b >> 3) & 0xF, mant = b & 0x7;
      float wv;
      if (exp == 0xF && mant == 0x7) {
        wv = as_type<float>(0x7fc00000u);           // qNaN -- matches quant.h's e4m3_decode
      } else {
        float mag = (exp == 0) ? (float(mant) * 0.001953125f)                 // subnormal: mant*2^-9
                                : (1.0f + float(mant)*0.125f) * exp2(float(int(exp) - 7));
        wv = sign ? -mag : mag;
      }
      acc += wv * xr[i] * scl[i/128];
    }
  } else {                                          // f32
    device const float* wr = (device const float*)(w) + (long)o * I;
    device const float4* w4 = (device const float4*)wr;
    for (int c = slane; c < I8; c += 32) acc += dot(w4[2*c],x4[2*c]) + dot(w4[2*c+1],x4[2*c+1]);
    for (int i = I8*8 + slane; i < I; i += 32) acc += wr[i] * xr[i];
  }
  acc = simd_sum(acc);
  // fmt==4 (per-group) and fmt==8 (per-block) already folded their scale into acc
  // above -- do not scale again.
  if (slane == 0) y[row] = (fmt == 4 || fmt == 8) ? acc : acc * scale[o];
}

// Batched bindless expert GEMV: each row gr belongs to expert erow[gr], whose weight and
// scale live at gpuAddresses waddr[e]/saddr[e] (zero-copy in the RAM slab). fmt 1=i8, 2=i4
// per-row, 4=i4 grouped (scale layout [O][ng], ng=ceil(K/qgs) -- same convention as mm_gemv
// fmt=4 above, but folded into the vectorized uchar4/float4 dot-product loop this kernel's
// fmt=2 branch already uses, since moe_gemv has no scalar strided branch to reuse).
// One SIMDGROUP per output row, 4 rows/threadgroup, 8-value loads: measured 1.5-2.1x over
// one-threadgroup-per-row with uchar2 loads (358-389 GB/s on engine-like block shapes).
kernel void moe_gemv(device const ulong* waddr [[buffer(0)]], device const ulong* saddr [[buffer(1)]],
                     device const int* erow [[buffer(2)]], device const float* xin [[buffer(3)]],
                     device float* yout [[buffer(4)]],
                     constant int& O [[buffer(5)]], constant int& K [[buffer(6)]],
                     constant int& Kin [[buffer(7)]], constant int& fmt [[buffer(8)]],
                     constant int& NT [[buffer(9)]], constant int& qgs [[buffer(10)]],
                     uint tg [[threadgroup_position_in_grid]],
                     uint slane [[thread_index_in_simdgroup]],
                     uint sgid [[simdgroup_index_in_threadgroup]]) {
  long row = (long)tg*4 + sgid; if (row >= NT) return;
  int gr = row / O, o = row % O; int e = erow[gr]; int K8 = (K & 7) ? 0 : (K/8);
  device const float* xr = xin + (long)gr * Kin;
  device const float* sc = (device const float*)(saddr[e]);
  device const float4* x4 = (device const float4*)xr;
  float acc = 0.0f;
  if (fmt == 2) { int rb=(K+1)/2; device const uchar* w=(device const uchar*)(waddr[e])+(long)o*rb;
    device const uchar4* w4=(device const uchar4*)w;
    for(int c=slane;c<K8;c+=32){ uchar4 b=w4[c];
      float4 w0=float4(float(int(b.x&0xF)-8),float(int(b.x>>4)-8),float(int(b.y&0xF)-8),float(int(b.y>>4)-8));
      float4 w1=float4(float(int(b.z&0xF)-8),float(int(b.z>>4)-8),float(int(b.w&0xF)-8),float(int(b.w>>4)-8));
      acc+=dot(w0,x4[2*c])+dot(w1,x4[2*c+1]); }
    for(int i=K8*8+slane;i<K;i+=32){ uchar b=w[i>>1]; int v=(i&1)?(b>>4):(b&0xF); acc+=float(v-8)*xr[i]; }
  } else if (fmt == 6) {                            // E8/IQ3: 98B per 256 weights, scales in-block
    long rb=((long)(K+255)/256)*98;                 // host guards K%256==0 (GLM dims are)
    device const uchar* w=(device const uchar*)(waddr[e])+(long)o*rb;
    int nsub=K/32;                                  // one 32-weight sub-block per lane step
    for(int s6=slane;s6<nsub;s6+=32){
      int b=s6>>3, ib=s6&7; device const uchar* blk=w+(long)b*98;
      uint word = uint(blk[64+ib*4]) | (uint(blk[65+ib*4])<<8)
                | (uint(blk[66+ib*4])<<16) | (uint(blk[67+ib*4])<<24);
      ushort dh = ushort(blk[96]) | (ushort(blk[97])<<8);
      float db = float(as_type<half>(dh)) * (0.5f + float((word>>28)&0xFu)) * 0.5f;
      device const uchar* idx = blk + ib*8;
      device const float4* xs = (device const float4*)(xr + s6*32);
      for(int l=0;l<4;l++){
        uint sv=(word>>(7*l))&0x7Fu;
        float4 m0=float4(E8G[idx[l*2+0]]), m1=float4(E8G[idx[l*2+1]]);
        float4 sA=float4((sv&1u)?-1.0f:1.0f,(sv&2u)?-1.0f:1.0f,(sv&4u)?-1.0f:1.0f,(sv&8u)?-1.0f:1.0f);
        float4 sB=float4((sv&16u)?-1.0f:1.0f,(sv&32u)?-1.0f:1.0f,(sv&64u)?-1.0f:1.0f,
                         (popcount(sv)&1u)?-1.0f:1.0f);   // j=7 closes by odd parity
        acc += (dot(m0*sA,xs[l*2]) + dot(m1*sB,xs[l*2+1])) * (0.5f*db);
      }
    }
  } else if (fmt == 5) {                            // bf16 raw rows: no scales, upper half of f32
    device const ushort* w=(device const ushort*)(waddr[e])+(long)o*K;
    device const ushort4* w4=(device const ushort4*)w;
    for(int c=slane;c<K8;c+=32){ ushort4 a=w4[2*c], b=w4[2*c+1];
      float4 w0=float4(as_type<float>(uint(a.x)<<16), as_type<float>(uint(a.y)<<16),
                       as_type<float>(uint(a.z)<<16), as_type<float>(uint(a.w)<<16));
      float4 w1=float4(as_type<float>(uint(b.x)<<16), as_type<float>(uint(b.y)<<16),
                       as_type<float>(uint(b.z)<<16), as_type<float>(uint(b.w)<<16));
      acc+=dot(w0,x4[2*c])+dot(w1,x4[2*c+1]); }
    for(int i=K8*8+slane;i<K;i+=32) acc += as_type<float>(uint(w[i])<<16)*xr[i];
  } else if (fmt == 4) {                            // grouped int4: per-expert scale [O][ng]
    int rb=(K+1)/2, ng=(K+qgs-1)/qgs; device const uchar* w=(device const uchar*)(waddr[e])+(long)o*rb;
    device const float* sr=sc+(long)o*ng;           // grouped scales for this output row
    device const uchar4* w4=(device const uchar4*)w;
    for(int c=slane;c<K8;c+=32){ uchar4 b=w4[c];
      float4 w0=float4(float(int(b.x&0xF)-8),float(int(b.x>>4)-8),float(int(b.y&0xF)-8),float(int(b.y>>4)-8));
      float4 w1=float4(float(int(b.z&0xF)-8),float(int(b.z>>4)-8),float(int(b.w&0xF)-8),float(int(b.w>>4)-8));
      int g0=(8*c+0)/qgs,g1=(8*c+1)/qgs,g2=(8*c+2)/qgs,g3=(8*c+3)/qgs;
      int g4=(8*c+4)/qgs,g5=(8*c+5)/qgs,g6=(8*c+6)/qgs,g7=(8*c+7)/qgs;
      acc+=dot(w0*float4(sr[g0],sr[g1],sr[g2],sr[g3]),x4[2*c])
          +dot(w1*float4(sr[g4],sr[g5],sr[g6],sr[g7]),x4[2*c+1]); }
    for(int i=K8*8+slane;i<K;i+=32){ uchar b=w[i>>1]; int v=(i&1)?(b>>4):(b&0xF); acc+=float(v-8)*xr[i]*sr[i/qgs]; }
  } else { device const char* w=(device const char*)(waddr[e])+(long)o*K;
    device const char4* w4=(device const char4*)w;
    for(int c=slane;c<K8;c+=32) acc+=dot(float4(w4[2*c]),x4[2*c])+dot(float4(w4[2*c+1]),x4[2*c+1]);
    for(int i=K8*8+slane;i<K;i+=32) acc+=float(w[i])*xr[i];
  }
  acc=simd_sum(acc);
  if(slane==0) yout[row] = (fmt==4 || fmt==5 || fmt==6) ? acc : acc*sc[o];   // fmt4 grouped / fmt6 in-block scales folded; fmt5 none
}

// fmt=6 activation rotation for the GPU-resident down-projection input: one FWHT
// tile per dispatch (block-diagonal tiling and sign stream match quant.h e8_rot_rows;
// signs are regenerated host-side with e8_signs and passed in). One threadgroup per
// row; tile fits threadgroup memory (n <= 4096).
kernel void moe_fwht(device float* v [[buffer(0)]], device const uchar* signs [[buffer(1)]],
                     constant int& dim [[buffer(2)]], constant int& off [[buffer(3)]],
                     constant int& n [[buffer(4)]],
                     uint tg [[threadgroup_position_in_grid]],
                     uint t [[thread_position_in_threadgroup]],
                     uint nt [[threads_per_threadgroup]]) {
  threadgroup float sh[4096];
  device float* row = v + (long)tg*dim + off;
  for (int i=int(t);i<n;i+=int(nt)){ float x=row[i]; if((signs[i>>3]>>(i&7))&1u) x=-x; sh[i]=x; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int len=1;len<n;len<<=1){
    for (int j=int(t);j<n/2;j+=int(nt)){
      int blk=j/len, k=j%len, i0=blk*(len<<1)+k;
      float a=sh[i0], b=sh[i0+len]; sh[i0]=a+b; sh[i0+len]=a-b;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float s=rsqrt(float(n));
  for (int i=int(t);i<n;i+=int(nt)) row[i]=sh[i]*s;
}
kernel void moe_silu(device float* g [[buffer(0)]], device const float* u [[buffer(1)]],
                     uint i [[thread_position_in_grid]]) { float v=g[i]; g[i]=(v/(1.0f+exp(-v)))*u[i]; }

// ===== Fused decode attention (GLM-5.2 dims, S=1) =====
constant int A_HID=6144, A_H=64, A_QLORA=2048, A_KVL=512, A_NOPE=192, A_ROPE=64, A_VH=256;
constant int A_QH=256 /*nope+rope*/, A_ROWSH=448 /*nope+vh*/;
// per-row in-place RMSNorm: row = threadgroup index, x[row*n + i]. grid = nrows threadgroups.
kernel void a_rmsnorm(device float* x [[buffer(0)]], device const float* w [[buffer(1)]],
                      constant int& n [[buffer(2)]], constant float& eps [[buffer(3)]],
                      uint row [[threadgroup_position_in_grid]], uint lid [[thread_position_in_threadgroup]], uint tgsz [[threads_per_threadgroup]]) {
  device float* xr=x+(long)row*n; threadgroup float red[256];
  float s=0; for(int i=lid;i<n;i+=tgsz) s+=xr[i]*xr[i];
  red[lid]=s; threadgroup_barrier(mem_flags::mem_threadgroup);
  for(uint k=tgsz/2;k>0;k>>=1){ if(lid<k) red[lid]+=red[lid+k]; threadgroup_barrier(mem_flags::mem_threadgroup); }
  float r=rsqrt(red[0]/n+eps); threadgroup_barrier(mem_flags::mem_threadgroup);
  for(int i=lid;i<n;i+=tgsz) xr[i]=xr[i]*r*w[i];
}
// interleaved partial RoPE. vv = v + base + s*rowstride + h*headstride, pos = PB+s. grid = S*nheads*(ROPE/2).
kernel void a_rope(device float* v [[buffer(0)]], constant int& base [[buffer(1)]],
                   constant int& rowstride [[buffer(2)]], constant int& headstride [[buffer(3)]],
                   constant int& nheads [[buffer(4)]], constant int& PB [[buffer(5)]],
                   constant float& theta [[buffer(6)]], uint gid [[thread_position_in_grid]]) {
  int hlf=A_ROPE/2; int idx=gid/hlf, j=gid%hlf; int s=idx/nheads, h=idx%nheads; int pos=PB+s;
  device float* vv=v+(long)base+(long)s*rowstride+(long)h*headstride;
  float inv=pow(theta, -2.0f*j/A_ROPE); float ang=pos*inv, cs=cos(ang), sn=sin(ang);
  float a=vv[2*j], b=vv[2*j+1]; vv[j]=a*cs-b*sn; vv[hlf+j]=b*cs+a*sn;
}
// per-row copy: dst[s*dststride + i] = src[s*srcstride + off + i]. grid = S*n.
kernel void a_copy(device const float* src [[buffer(0)]], constant int& off [[buffer(1)]], constant int& srcstride [[buffer(2)]],
                   device float* dst [[buffer(3)]], constant int& dststride [[buffer(4)]], constant int& n [[buffer(5)]],
                   uint gid [[thread_position_in_grid]]) { int s=gid/n, i=gid%n; dst[(long)s*dststride+i]=src[(long)s*srcstride+off+i]; }
// ---- absorption core (S query rows, per-row causal). q:[S,H*QH]; qabs/clat:[S*H,KVL];
//      sc:[S*H,T]; ctx:[S*H,VH]. Query row s (abs pos PB+s) attends keys [0, PB+s]. ----
constant int A_QHH=A_H*A_QH;
// kv_b inline dequant of column i of output row `row`. fmt=2 -> one scale per row;
// fmt=4 -> grouped int4, one scale per gs-wide group along the A_KVL input dim
// (scale layout [O][ng], ng=ceil(A_KVL/gs)), matching QT fmt=4 / mm_gemv above.
inline float a_deqrow(device const uchar* base, int row, int i, device const float* sc, int fmt, int gs){
  device const uchar* w=base+(long)row*((A_KVL+1)/2); uchar b=w[i>>1]; int val=(i&1)?(b>>4):(b&0xF);
  float s = (fmt==4) ? sc[(long)row*((A_KVL+gs-1)/gs) + i/gs] : sc[row];
  return float(val-8)*s; }
kernel void a_qabs(device const uchar* kvb [[buffer(0)]], device const float* sc [[buffer(1)]],
                   device const float* q [[buffer(2)]], device float* qabs [[buffer(3)]],
                   constant int& fmt [[buffer(4)]], constant int& gs [[buffer(5)]],
                   uint gid [[thread_position_in_grid]]) {
  int s=gid/(A_H*A_KVL), r=gid%(A_H*A_KVL), h=r/A_KVL, i=r%A_KVL; int rbase=h*A_ROWSH;
  device const float* qp=q+(long)s*A_QHH+(long)h*A_QH;
  float a=0; for(int d=0;d<A_NOPE;d++) a+=qp[d]*a_deqrow(kvb,rbase+d,i,sc,fmt,gs); qabs[(long)(s*A_H+h)*A_KVL+i]=a;
}
kernel void a_score(device const float* qabs [[buffer(0)]], device const float* Lc [[buffer(1)]],
                    device const float* Rc [[buffer(2)]], device const float* q [[buffer(3)]],
                    device float* sc [[buffer(4)]], constant int& T [[buffer(5)]], constant float& ascale [[buffer(6)]],
                    constant int& PB [[buffer(7)]], uint gid [[thread_position_in_grid]]) {
  int s=gid/(A_H*T), r=gid%(A_H*T), h=r/T, t=r%T; long o=(long)(s*A_H+h)*T+t;
  if(t > PB+s){ sc[o]=-1e30f; return; }                                 // causal mask
  device const float* qa=qabs+(long)(s*A_H+h)*A_KVL; device const float* Lt=Lc+(long)t*A_KVL;
  device const float* qr=q+(long)s*A_QHH+(long)h*A_QH+A_NOPE; device const float* Rt=Rc+(long)t*A_ROPE;
  float a=0; for(int i=0;i<A_KVL;i++) a+=qa[i]*Lt[i]; for(int d=0;d<A_ROPE;d++) a+=qr[d]*Rt[d]; sc[o]=a*ascale;
}
kernel void a_smax(device float* sc [[buffer(0)]], constant int& T [[buffer(1)]],
                   uint sh [[threadgroup_position_in_grid]], uint lid [[thread_position_in_threadgroup]], uint tgsz [[threads_per_threadgroup]]) {
  device float* s=sc+(long)sh*T; threadgroup float red[256];
  float m=-1e30f; for(int t=lid;t<T;t+=tgsz) m=max(m,s[t]); red[lid]=m; threadgroup_barrier(mem_flags::mem_threadgroup);
  for(uint k=tgsz/2;k>0;k>>=1){ if(lid<k) red[lid]=max(red[lid],red[lid+k]); threadgroup_barrier(mem_flags::mem_threadgroup);}
  float mx=red[0]; threadgroup_barrier(mem_flags::mem_threadgroup);
  float sum=0; for(int t=lid;t<T;t+=tgsz){ float e=exp(s[t]-mx); s[t]=e; sum+=e; } red[lid]=sum; threadgroup_barrier(mem_flags::mem_threadgroup);
  for(uint k=tgsz/2;k>0;k>>=1){ if(lid<k) red[lid]+=red[lid+k]; threadgroup_barrier(mem_flags::mem_threadgroup);}
  float tot=red[0]; threadgroup_barrier(mem_flags::mem_threadgroup); for(int t=lid;t<T;t+=tgsz) s[t]/=tot;
}
kernel void a_clat(device const float* sc [[buffer(0)]], device const float* Lc [[buffer(1)]],
                   device float* clat [[buffer(2)]], constant int& T [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
  int sh=gid/A_KVL, i=gid%A_KVL; device const float* s=sc+(long)sh*T; float a=0;
  for(int t=0;t<T;t++) a+=s[t]*Lc[(long)t*A_KVL+i]; clat[(long)sh*A_KVL+i]=a;
}
kernel void a_ctx(device const uchar* kvb [[buffer(0)]], device const float* sc [[buffer(1)]],
                  device const float* clat [[buffer(2)]], device float* ctx [[buffer(3)]],
                  constant int& fmt [[buffer(4)]], constant int& gs [[buffer(5)]],
                  uint gid [[thread_position_in_grid]]) {
  int sh=gid/A_VH, j=gid%A_VH, h=sh%A_H; int row=h*A_ROWSH+A_NOPE+j; device const float* cl=clat+(long)sh*A_KVL;
  float a=0; for(int i=0;i<A_KVL;i++) a+=cl[i]*a_deqrow(kvb,row,i,sc,fmt,gs); ctx[(long)sh*A_VH+j]=a;
}

// ===== full-layer tail kernels =====
// y[i] += a[i]  (residual add), grid = n
kernel void a_add(device float* y [[buffer(0)]], device const float* a [[buffer(1)]],
                  uint i [[thread_position_in_grid]]) { y[i] += a[i]; }
// router: logit[s][e] = x[s].w_e (f32 rows [E,D]) -> sig=1/(1+exp(-logit)). One simdgroup/row.
kernel void r_router(device const float* rw [[buffer(0)]], device const float* x [[buffer(1)]],
                     device float* sig [[buffer(2)]], constant int& E [[buffer(3)]],
                     constant int& D [[buffer(4)]], constant int& NT [[buffer(5)]],
                     uint tg [[threadgroup_position_in_grid]],
                     uint slane [[thread_index_in_simdgroup]], uint sgid [[simdgroup_index_in_threadgroup]]) {
  long row=(long)tg*4+sgid; if(row>=NT) return;
  int e=row%E, s=row/E;
  device const float4* w4=(device const float4*)(rw+(long)e*D);
  device const float4* x4=(device const float4*)(x+(long)s*D);
  float acc=0; int D4=D/4;
  for(int c=slane;c<D4;c+=32) acc+=dot(w4[c],x4[c]);
  acc=simd_sum(acc);
  if(slane==0) sig[row]=1.0f/(1.0f+exp(-acc));
}
// exact replica of glm.c phase-A selection per row s (serial, deterministic ties):
// choice=sig+bias; greedy top-Ksel by choice; w=sig[best]; optional topp truncation
// (insertion-sort desc + cumulative); optional norm_topk; * routed_scale.
kernel void r_top8(device const float* sig [[buffer(0)]], device const float* bias [[buffer(1)]],
                   device int* idx [[buffer(2)]], device float* w [[buffer(3)]],
                   device int* keff [[buffer(4)]], constant int& E [[buffer(5)]],
                   constant int& K [[buffer(6)]], constant int& Ksel [[buffer(7)]],
                   constant float& topp [[buffer(8)]], constant int& normk [[buffer(9)]],
                   constant float& rscale [[buffer(10)]],
                   uint s [[thread_position_in_grid]]) {
  device const float* sg=sig+(long)s*E;
  device int* id_=idx+(long)s*K; device float* ww=w+(long)s*K;
  for(int kk=0;kk<Ksel;kk++){ int best=-1; float bv=-1e30f;
    for(int e=0;e<E;e++){ bool tk=false; for(int j=0;j<kk;j++) if(id_[j]==e){tk=true;break;}
      float ch=sg[e]+bias[e];
      if(!tk && ch>bv){bv=ch;best=e;} }
    id_[kk]=best; ww[kk]=sg[best];
  }
  int Ke=Ksel;
  if(topp>0.0f && topp<1.0f){
    for(int a=1;a<Ksel;a++){ int ii=id_[a]; float wv=ww[a]; int b=a-1;
      while(b>=0 && ww[b]<wv){ ww[b+1]=ww[b]; id_[b+1]=id_[b]; b--; } ww[b+1]=wv; id_[b+1]=ii; }
    float tot=1e-20f; for(int kk=0;kk<Ksel;kk++) tot+=ww[kk];
    float cum=0; for(int kk=0;kk<Ksel;kk++){ cum+=ww[kk]; if(cum>=topp*tot){ Ke=kk+1; break; } }
  }
  keff[s]=Ke;
  if(normk){ float sm=0; for(int kk=0;kk<Ke;kk++) sm+=ww[kk]; sm+=1e-20f; for(int kk=0;kk<Ke;kk++) ww[kk]/=sm; }
  for(int kk=0;kk<Ke;kk++) ww[kk]*=rscale;
}
// parallel replica of r_top8's selection on ONE SIMDGROUP per row instead of one serial
// thread (bench/kernels @ 27bfe83: serial r_top8 measured 0.465 ms/layer, ~55% of the
// layer CB; this replica ~93x faster with exactly matching output). EXACT-MATCH is the
// contract: each lane owns ceil(E/32) contiguous experts (blocked) and keeps a taken
// bitmask; per selection step: lane-local strict-'>' ascending max (lowest index wins
// within a lane, matching the serial ascending scan), then a shuffle-down argmax
// reduction where ties prefer the LOWER index — together exactly the serial kernel's
// first-max-wins order. The topp/normk/rscale tail is the serial code verbatim on lane 0
// (same ops, same order => bitwise-identical results; metal-test enforces this with
// memcmp). Contract: E<=256 (ch[8]/taken mask sizing: ceil(E/32)<=8) — the defensive
// return below makes an out-of-contract dispatch a visible no-op (idx/w/keff untouched),
// never an OOB write; both call sites (coli_metal_layer_decode's dispatch and the
// standalone coli_metal_rtop8 runner) additionally gate on E<=256 in host code before
// selecting this pipeline at all, so the return here is defense-in-depth, not the only
// guard. Sentinel-per-lane design (ch[j]=-1e30f for e>=E) makes non-multiple-of-32 E
// and small E correct without special-casing — validated for E=24, E=168 (REAP
// expert-pruned packages, see the upstream feature-request thread) and E=256 by metal-test.
// ASSUMES SIMD width 32 (shuffle offsets 16..1, 32-thread threadgroup per row): enforced
// at init — coli_metal_init clears g_rtop8_width_ok (and therefore both call sites' use
// of this pipeline) if threadExecutionWidth != 32.
kernel void r_top8_par(device const float* sig [[buffer(0)]], device const float* bias [[buffer(1)]],
                       device int* idx [[buffer(2)]], device float* w [[buffer(3)]],
                       device int* keff [[buffer(4)]], constant int& E [[buffer(5)]],
                       constant int& K [[buffer(6)]], constant int& Ksel [[buffer(7)]],
                       constant float& topp [[buffer(8)]], constant int& normk [[buffer(9)]],
                       constant float& rscale [[buffer(10)]],
                       uint s [[threadgroup_position_in_grid]],
                       uint slane [[thread_index_in_simdgroup]]) {
  if(E>256) return;
  device const float* sg=sig+(long)s*E;
  device int* id_=idx+(long)s*K; device float* ww=w+(long)s*K;
  int per=(E+31)/32, base=(int)slane*per;
  float ch[8]; uint taken=0u;
  for(int j=0;j<per;j++){ int e=base+j; ch[j]=(e<E)?sg[e]+bias[e]:-1e30f; }
  for(int kk=0;kk<Ksel;kk++){
    float bv=-1e30f; int bi=0x7FFFFFFF;
    for(int j=0;j<per;j++) if(!(taken&(1u<<j)) && ch[j]>bv){ bv=ch[j]; bi=base+j; }
    for(uint off=16;off>0;off>>=1){
      float ov=simd_shuffle_down(bv,off); int oi=simd_shuffle_down(bi,off);
      if(ov>bv || (ov==bv && oi<bi)){ bv=ov; bi=oi; }
    }
    bv=simd_broadcast(bv,0); bi=simd_broadcast(bi,0);
    if(bi>=base && bi<base+per) taken|=1u<<(bi-base);
    if(slane==0){ id_[kk]=bi; ww[kk]=sg[bi]; }
  }
  if(slane!=0) return;
  int Ke=Ksel;
  if(topp>0.0f && topp<1.0f){
    for(int a=1;a<Ksel;a++){ int ii=id_[a]; float wv=ww[a]; int b=a-1;
      while(b>=0 && ww[b]<wv){ ww[b+1]=ww[b]; id_[b+1]=id_[b]; b--; } ww[b+1]=wv; id_[b+1]=ii; }
    float tot=1e-20f; for(int kk=0;kk<Ksel;kk++) tot+=ww[kk];
    float cum=0; for(int kk=0;kk<Ksel;kk++){ cum+=ww[kk]; if(cum>=topp*tot){ Ke=kk+1; break; } }
  }
  keff[s]=Ke;
  if(normk){ float sm=0; for(int kk=0;kk<Ke;kk++) sm+=ww[kk]; sm+=1e-20f; for(int kk=0;kk<Ke;kk++) ww[kk]/=sm; }
  for(int kk=0;kk<Ke;kk++) ww[kk]*=rscale;
}

// ===== KV cache kernels =====
// per-row rope_interleave on a single qk_rope vector (no heads). grid = S*(qk_rope/2).
kernel void kv_rope(device float* R [[buffer(0)]], constant int& pos_base [[buffer(1)]],
                    constant int& qk_rope [[buffer(2)]], constant float& theta [[buffer(3)]],
                    uint gid [[thread_position_in_grid]]) {
  int S=gid/(qk_rope/2); int j=gid%(qk_rope/2); int s=S; int hl=qk_rope/2;
  int pos=pos_base+s;
  device float* rv=R+(long)s*qk_rope;
  float inv=pow(theta, -2.0f*j/(float)qk_rope); float ang=pos*inv, cs=cos(ang), sn=sin(ang);
  // Read original pair BEFORE any write (threadgroup-local copy to avoid hazard)
  thread float a=rv[2*j], b=rv[2*j+1];
  // Write to both halves — may read from positions we'll also write, but we already have a/b
  rv[j]     = a*cs - b*sn;
  rv[hl+j]  = b*cs + a*sn;
}
// clear range: zero rows [from, to) [from,to) of Lc/Rc. grid = 2*nrows*maxD.
kernel void kv_clear(device float* Lc [[buffer(0)]], device float* Rc [[buffer(1)]],
                     constant int& from [[buffer(2)]], constant int& to [[buffer(3)]],
                     constant int& kv_lora [[buffer(4)]], constant int& qk_rope [[buffer(5)]],
                     uint gid [[thread_position_in_grid]]) {
  int nr=to-from; int sel=gid%(2*nr), row=gid/(2*nr);
   int s= row; if(s>=nr) return;
   if(sel < nr){ Lc[(long)(from+s)*kv_lora+gid%(kv_lora)]=0.f; }
   else        { Rc[(long)(from+s)*qk_rope+gid%(qk_rope)]=0.f; }
}

// ===== KDA Conv1d + SiLU kernel =====
// Each thread (0..P-1) handles one dimension of one projection (q, k, or v).
//  conv_win[d*K]   — stateful shift register [P*K], oldest first, newest at K-1
//  vec[d]          — raw projected value (saved to window[K-1], overwritten with SiLU result)
//  taps[d*K]       — constant convolution taps [P*K]
kernel void kda_conv_silu(
    device float* conv_win  [[buffer(0)]],
    device float* vec       [[buffer(1)]],
    device const float* taps [[buffer(2)]],
    constant int& P         [[buffer(3)]],
    constant int& K         [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
  int d = (int)gid;
  if (d >= P) return;
  long base = (long)d * K;

  // Shift window left by 1 and store new value at position K-1
  for (int j = 0; j < K - 1; j++) {
    conv_win[base + j] = conv_win[base + j + 1];
  }
  conv_win[base + K - 1] = vec[d];

  // Compute dot product of taps and window
  float acc = 0.f;
  for (int j = 0; j < K; j++) {
    acc += taps[base + j] * conv_win[base + j];
  }

  // Apply SiLU: x / (1 + exp(-x))
  vec[d] = acc / (1.f + exp(-acc));
}

// ===== KDA L2 normalization kernel =====
// Each thread (0..H-1) normalizes one head of Q and K in-place.
//  q[h*hd:(h+1)*hd] *= 1/sqrt(sum(q^2) + eps) * qscale
//  k[h*hd:(h+1)*hd] *= 1/sqrt(sum(k^2) + eps)
kernel void kda_l2_norm(
    device float* q       [[buffer(0)]],
    device float* k       [[buffer(1)]],
    constant int& H       [[buffer(2)]],
    constant int& hd      [[buffer(3)]],
    constant float& qscale [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
  int h = (int)gid;
  if (h >= H) return;

  long base = (long)h * hd;
  float sq = 0.f, sk = 0.f;
  for (int i = 0; i < hd; i++) {
    sq += q[base + i] * q[base + i];
    sk += k[base + i] * k[base + i];
  }
  sq = 1.f / sqrt(sq + 1e-6f);
  sk = 1.f / sqrt(sk + 1e-6f);
  float qs = sq * qscale;
  for (int i = 0; i < hd; i++) {
    q[base + i] *= qs;
    k[base + i] *= sk;
  }
}

// ===== KDA state recurrence kernel =====
// Each thread handles one (h, i) element of output space [H, hd].
// Two-pass sweep: (1) decay S by alpha, accumulate kS = S^T * kn,
//   then vt = (vh - kS) * beta; (2) expand S += kn[kk]*vt[vv],
//   merge oh[i] += qn[kk] * S[kk][i].
kernel void kda_state(
    device float* S      [[buffer(0)]],     // [H, hd, hd] in-place state
    device const float* qn     [[buffer(1)]],   // [H, hd] L2-normalized, gated Q
    device const float* kn     [[buffer(2)]],   // [H, hd] L2-normalized K
    device const float* vh     [[buffer(3)]],   // [H, hd] gated V
    device const float* alpha  [[buffer(4)]],   // [H, hd] per-head decay factors
    device const float* beta   [[buffer(5)]],   // [H] mixing factor
    device float* oh     [[buffer(6)]],     // [H, hd] output accumulator
    constant int& H      [[buffer(7)]],
    constant int& hd     [[buffer(8)]],
    uint gid [[thread_position_in_grid]]) {
  int th = gid / hd;
  int i = gid % hd;
  if (th >= H) return;
  long hS = (long)th * hd * hd;
  long hq = (long)th * hd;
  const device float* qn_h = qn + hq;
  const device float* kn_h = kn + hq;
  const device float* vh_h = vh + hq;
  const device float* alpha_h = alpha + hq;
  // Pass 1: decay + kS accumulate for this output column
  float kiS = 0.f;
  for (int kk = 0; kk < hd; kk++) {
    long row = hS + (long)kk * hd;
    float al = alpha_h[kk];
    S[row + i] *= al;
    kiS += kn_h[kk] * S[row + i];
  }
  float vi = (vh_h[i] - kiS) * beta[th];
  // Pass 2: expand + merge
  float oi = 0.f;
  for (int kk = 0; kk < hd; kk++) {
    long row = hS + (long)kk * hd;
    float kv = kn_h[kk];
    S[row + i] += kv * vi;
    oi += qn_h[kk] * S[row + i];
  }
  oh[hq + i] = oi;
}
