223-    uint8_t e; // E8M0
224-    uint8_t qs[QK_MXFP6*6/8]; // E2M3, 6-bit packed codes, value j at bit 6*j (little-endian bit order)
225:} block_mxfp6;
226:static_assert(sizeof(block_mxfp6) == sizeof(uint8_t) + QK_MXFP6*6/8, "wrong mxfp6 block size/padding");
227-
228-#define QK_MXFP8 32
229-typedef struct {
230-    uint8_t e; // E8M0
231-    uint8_t qs[QK_MXFP8]; // E4M3
232:} block_mxfp8;
233:static_assert(sizeof(block_mxfp8) == sizeof(uint8_t) + QK_MXFP8, "wrong mxfp8 block size/padding");
234-
235-#define QK_NVFP4 64
236-#define QK_NVFP4_SUB 16  // sub-block size for per-group scales
237-typedef struct {
238-    uint8_t d[QK_NVFP4/QK_NVFP4_SUB]; // UE4M3 scales (4 bytes, one per 16-element sub-block)
239-    uint8_t qs[QK_NVFP4/2];           // packed 4-bit E2M1 values (32 bytes)
240-} block_nvfp4;
241-static_assert(sizeof(block_nvfp4) == sizeof(uint8_t)*(QK_NVFP4/QK_NVFP4_SUB) + QK_NVFP4/2, "wrong nvfp4 block size/padding");
242-
243-#define QK5_0 32
244-typedef struct {
245-    ggml_half d;           // delta
