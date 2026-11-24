window.MUJOCO_OVERVIEW_STATIC_LAYOUT = true;
// Browser bootstrap and asset prefetcher for the MuJoCo Playground homepage.

(() => {
  const loadingOverlay = document.getElementById('loadingOverlay');

  /**
   * Returns true on phones and tablets with constrained browser memory budgets.
   * @return {boolean}
   */
  function isMemoryConstrainedDevice() {
    const nav = typeof navigator !== 'undefined' ? navigator : null;
    if (nav && nav.userAgentData &&
        typeof nav.userAgentData.mobile === 'boolean' &&
        nav.userAgentData.mobile) {
      return true;
    }
    const ua = (nav && nav.userAgent) || '';
    if (/iPad|iPhone|iPod|Android|Mobile|Windows Phone|IEMobile|Opera Mini/i
            .test(ua)) {
      return true;
    }
    if (nav && nav.platform === 'MacIntel' && nav.maxTouchPoints > 1) {
      return true;
    }
    const coarseOrNoHover = Boolean(
        window.matchMedia &&
        (window.matchMedia('(hover: none)').matches ||
         window.matchMedia('(pointer: coarse)').matches));
    const minDim = Math.min(
        (window.screen && window.screen.width) || window.innerWidth || 0,
        (window.screen && window.screen.height) || window.innerHeight || 0);
    return coarseOrNoHover && minDim > 0 && minDim < 768;
  }
  const IS_MEMORY_CONSTRAINED = isMemoryConstrainedDevice();

  /**
   * Pre-allocates WebAssembly linear memory (paired with `-sIMPORTED_MEMORY=1`
   * and `-sINITIAL_MEMORY=64mb`) so desktop browsers can grow up to 4 GiB
   * while mobile browsers stay safely within their per-tab memory ceilings.
   * @return {!WebAssembly.Memory|undefined}
   */
  function createWasmMemory() {
    const mbPages = 16;  // 1 MiB = 16 * 64 KiB WebAssembly pages.
    const candidates = IS_MEMORY_CONSTRAINED ?
        [
          {initial: 128 * mbPages, maximum: 768 * mbPages},
          {initial: 128 * mbPages, maximum: 512 * mbPages},
          {initial: 96 * mbPages, maximum: 384 * mbPages},
          {initial: 64 * mbPages, maximum: 256 * mbPages},
        ] :
        [
          {initial: 3072 * mbPages, maximum: 65536},
          {initial: 1024 * mbPages, maximum: 65536},
          {initial: 256 * mbPages, maximum: 65536},
          {initial: 64 * mbPages, maximum: 65536},
        ];
    for (const spec of candidates) {
      try {
        return new WebAssembly.Memory(spec);
      } catch (err) {
        console.warn('[wasm] WebAssembly.Memory fallback:', spec, err);
      }
    }
    return undefined;
  }

  /** Hides the full-screen loading overlay spinner. */
  function hideLoading() {
    if (loadingOverlay) loadingOverlay.style.display = 'none';
  }

  /**
   * Returns the directory URL hosting the current page without trailing slash.
   * @return {string}
   */
  function getBaseUrl() {
    return window.location.origin +
        window.location.pathname.substring(
            0, window.location.pathname.lastIndexOf('/'));
  }

  /**
   * Decodes a base64-encoded payload into a fresh Uint8Array.
   * @param {string} b64
   * @return {!Uint8Array}
   */
  function decodeBase64Bytes(b64) {
    const binStr = atob(b64);
    const len = binStr.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; ++i) {
      bytes[i] = binStr.charCodeAt(i);
    }
    return bytes;
  }

  const primedUrls = new Set();

  /**
   * Registers pre-fetched bytes for `httpUrl` in the WASM FetchCache once.
   * @param {string} httpUrl
   * @param {?Uint8Array} bytes
   */
  function primeUrlBytes(httpUrl, bytes) {
    if (primedUrls.has(httpUrl) || !bytes) return;
    primedUrls.add(httpUrl);
    moduleConfig.primeFetchCache(httpUrl, bytes);
  }

  /**
   * Pushes all CSP-bundled model/policy files from `models_bundle.js` into
   * FetchCache.
   * @param {string} modelBaseUrl
   * @param {function({files: number, bytes: number, errors: number})=}
   *     onProgress
   * @return {?{files: number, bytes: number, errors: number}}
   */
  function primeAllBundledModels(modelBaseUrl, onProgress) {
    const bundle = window.__MUJOCO_OVERVIEW_MODEL_FILES__;
    if (!bundle) return null;
    const prefix =
        modelBaseUrl.endsWith('/') ? modelBaseUrl : modelBaseUrl + '/';
    const stats = {files: 0, bytes: 0, errors: 0};
    for (const key of Object.keys(bundle)) {
      const bytes = decodeBase64Bytes(bundle[key]);
      delete bundle[key];
      primeUrlBytes(prefix + key, bytes);
      stats.files++;
      stats.bytes += bytes.length;
    }
    delete window.__MUJOCO_OVERVIEW_MODEL_FILES__;
    onProgress?.(stats);
    return stats;
  }

  /**
   * Fetches raw bytes for `httpUrl` from bundled JS assets or HTTP `fetch()`.
   * @param {string} httpUrl
   * @return {!Promise<{ok: boolean, bytes: ?Uint8Array}>}
   */
  async function fetchUrlBytes(httpUrl) {
    if (primedUrls.has(httpUrl)) {
      return {ok: true, bytes: null};
    }
    if (window.__MUJOCO_OVERVIEW_MODEL_FILES__) {
      const marker = '/model/';
      const idx = httpUrl.indexOf(marker);
      if (idx !== -1) {
        const relPath = decodeURIComponent(
            httpUrl.substring(idx + marker.length).split('?')[0]);
        const b64 = window.__MUJOCO_OVERVIEW_MODEL_FILES__[relPath];
        if (b64 !== undefined) {
          const bytes = decodeBase64Bytes(b64);
          delete window.__MUJOCO_OVERVIEW_MODEL_FILES__[relPath];
          primeUrlBytes(httpUrl, bytes);
          return {ok: true, bytes};
        }
      }
    }
    try {
      const resp = await fetch(httpUrl);
      if (!resp.ok) {
        console.warn('[prefetch] HTTP', resp.status, httpUrl);
        return {ok: false, bytes: null};
      }
      const bytes = new Uint8Array(await resp.arrayBuffer());
      return {ok: true, bytes};
    } catch (err) {
      console.warn('[prefetch]', httpUrl, err);
      return {ok: false, bytes: null};
    }
  }

  /**
   * Returns true when `bytes` is empty/uninspected or begins with `'<'` after
   * an optional UTF-8 BOM and ASCII whitespace.
   * @param {?Uint8Array} bytes
   * @return {boolean}
   */
  function looksLikeXmlBytes(bytes) {
    if (!bytes || bytes.length === 0) return true;
    let i = 0;
    if (bytes.length >= 3 && bytes[0] === 0xef && bytes[1] === 0xbb &&
        bytes[2] === 0xbf) {
      i = 3;
    }
    while (i < bytes.length &&
           (bytes[i] === 0x20 || bytes[i] === 0x09 || bytes[i] === 0x0a ||
            bytes[i] === 0x0d)) {
      ++i;
    }
    return i < bytes.length && bytes[i] === 0x3c;
  }

  /**
   * Prefetches a root MJCF XML and all its transitive external dependencies.
   * @param {string} rootUrl
   * @param {function({files: number, bytes: number, errors: number})=}
   *     onProgress
   * @return {!Promise<{files: number, bytes: number, errors: number}>}
   */
  async function prefetchModelAssets(rootUrl, onProgress) {
    if (!rootUrl || !/^https?:/.test(rootUrl)) {
      return {files: 0, bytes: 0, errors: 0};
    }
    const stats = {files: 0, bytes: 0, errors: 0};
    const rootRes = await fetchUrlBytes(rootUrl);
    if (!rootRes.ok) {
      stats.errors++;
      return stats;
    }
    if (rootRes.bytes) {
      primeUrlBytes(rootUrl, rootRes.bytes);
      stats.files++;
      stats.bytes += rootRes.bytes.length;
      onProgress?.(stats);
    }

    if (!looksLikeXmlBytes(rootRes.bytes)) {
      return stats;
    }

    try {
      const depsVec = await moduleConfig.getXMLDependencies(rootUrl);
      if (depsVec) {
        const deps = [];
        for (let i = 0; i < depsVec.size(); ++i) {
          deps.push(depsVec.get(i));
        }
        depsVec.delete();
        const depResults = await Promise.all(deps.map(async (depUrl) => {
          const res = await fetchUrlBytes(depUrl);
          return {depUrl, ...res};
        }));
        for (const item of depResults) {
          if (!item.ok) {
            stats.errors++;
          } else if (item.bytes) {
            primeUrlBytes(item.depUrl, item.bytes);
            stats.files++;
            stats.bytes += item.bytes.length;
            onProgress?.(stats);
          }
        }
      }
    } catch (e) {
      console.log('[prefetch] Self-contained or non-XML model:', rootUrl);
    }
    return stats;
  }

  /**
   * Prefetches a standalone binary asset (such as `turkey_policy.onnx`) into
   * the WASM FetchCache.
   * @param {string} url
   * @param {function({files: number, bytes: number, errors: number})=}
   *     onProgress
   * @return {!Promise<{files: number, bytes: number, errors: number}>}
   */
  async function prefetchBinaryAsset(url, onProgress) {
    const stats = {files: 0, bytes: 0, errors: 0};
    if (!url || !/^https?:/.test(url)) return stats;
    const res = await fetchUrlBytes(url);
    if (!res.ok) {
      stats.errors++;
      return stats;
    }
    if (res.bytes) {
      primeUrlBytes(url, res.bytes);
      stats.files++;
      stats.bytes += res.bytes.length;
      onProgress?.(stats);
    }
    return stats;
  }

  const moduleConfig = {
    wasmMemory: createWasmMemory(),
    preRun: [],
    postRun: [],
    // Satisfy Blaze rules_wasm's auto-generated MEMFS file_packager without a
    // redundant network request; WASM loads Filament and font assets via
    // Module.registerAsset() -> AssetRegistry instead of MEMFS.
    getPreloadedPackage(_remotePackageName, remotePackageSize) {
      return new ArrayBuffer(remotePackageSize);
    },
    locateFile(path) {
      const baseURL = getBaseUrl();
      if (path.endsWith('.wasm')) {
        return baseURL + '/bin/mujoco_homepage.wasm';
      }
      return baseURL + '/bin/' + path;
    },
    print: console.log,
    printErr: (text) => {
      if (typeof text === 'string' &&
          (text.startsWith('I0000 ') ||
           text.startsWith('WARNING: Logging before InitGoogle()') ||
           text.startsWith('Rules for selecting event targets in HTML5 API'))) {
        console.log(text);
        return;
      }
      console.error(text + '\n' + new Error().stack);
    },
    canvas: (() => {
      const canvas = document.getElementById('canvas');
      canvas.addEventListener('webglcontextlost', (e) => {
        alert('WebGL context lost. You will need to reload the page.');
        e.preventDefault();
      }, false);
      canvas.addEventListener('contextmenu', (e) => {
        e.preventDefault();
      }, false);
      const preventBrowserZoom = (e) => {
        e.preventDefault();
      };
      canvas.addEventListener(
          'gesturestart', preventBrowserZoom, {passive: false});
      canvas.addEventListener(
          'gesturechange', preventBrowserZoom, {passive: false});
      canvas.addEventListener(
          'gestureend', preventBrowserZoom, {passive: false});
      canvas.addEventListener('touchmove', (e) => {
        if (e.touches.length > 1) e.preventDefault();
      }, {passive: false});
      return canvas;
    })(),
    setStatus() {},
    totalDependencies: 0,
    monitorRunDependencies() {},
    onRuntimeInitialized: () => {
      const assetsToPrefetch = window.MUJOCO_OVERVIEW_STATIC_LAYOUT ?
          [
            'assets/fontawesome-webfont.ttf',
            'assets/ibl.ktx',
            'assets/AtkinsonHyperlegibleNext[wght].ttf',
            'assets/AtkinsonHyperlegibleMono-Regular.ttf',
            'assets/pbr.filamat',
            'assets/pbr_reflect.filamat',
            'assets/pbr_packed_reflect.filamat',
            'assets/pbr_transparent.filamat',
            'assets/pbr_packed.filamat',
            'assets/pbr_packed_transparent.filamat',
            'assets/phong_2d_fade.filamat',
            'assets/phong_2d.filamat',
            'assets/phong_2d_uv_fade.filamat',
            'assets/phong_2d_uv.filamat',
            'assets/phong_2d_uv_reflect.filamat',
            'assets/phong_color_fade.filamat',
            'assets/phong_color.filamat',
            'assets/phong_color_reflect.filamat',
            'assets/phong_cube_fade.filamat',
            'assets/phong_cube.filamat',
            'assets/phong_cube_reflect.filamat',
            'assets/outline_composite.filamat',
            'assets/outline_flatten.filamat',
            'assets/outline_jumpflood.filamat',
            'assets/decor.filamat',
            'assets/decor_fade.filamat',
            'assets/haze.filamat',
            'assets/unlit_depth.filamat',
            'assets/unlit_segmentation.filamat',
            'assets/unlit_ui.filamat',
          ] :
          [
            '/third_party/mujoco/src/render/filament/assets/ibl.ktx',
            '/third_party/mujoco/src/render/filament/assets/pbr.filamat',
            '/third_party/mujoco/src/render/filament/assets/pbr_reflect.filamat',
            '/third_party/mujoco/src/render/filament/assets/pbr_packed_reflect.filamat',
            '/third_party/mujoco/src/render/filament/assets/pbr_transparent.filamat',
            '/third_party/mujoco/src/render/filament/assets/pbr_packed.filamat',
            '/third_party/mujoco/src/render/filament/assets/pbr_packed_transparent.filamat',
            '/third_party/mujoco/src/render/filament/assets/phong_2d.filamat',
            '/third_party/mujoco/src/render/filament/assets/phong_2d_fade.filamat',
            '/third_party/mujoco/src/render/filament/assets/phong_2d_uv.filamat',
            '/third_party/mujoco/src/render/filament/assets/phong_2d_uv_fade.filamat',
            '/third_party/mujoco/src/render/filament/assets/phong_2d_uv_reflect.filamat',
            '/third_party/mujoco/src/render/filament/assets/phong_color.filamat',
            '/third_party/mujoco/src/render/filament/assets/phong_color_fade.filamat',
            '/third_party/mujoco/src/render/filament/assets/phong_color_reflect.filamat',
            '/third_party/mujoco/src/render/filament/assets/phong_cube.filamat',
            '/third_party/mujoco/src/render/filament/assets/phong_cube_fade.filamat',
            '/third_party/mujoco/src/render/filament/assets/phong_cube_reflect.filamat',
            '/third_party/mujoco/src/render/filament/assets/outline_composite.filamat',
            '/third_party/mujoco/src/render/filament/assets/outline_flatten.filamat',
            '/third_party/mujoco/src/render/filament/assets/outline_jumpflood.filamat',
            '/third_party/mujoco/src/render/filament/assets/decor.filamat',
            '/third_party/mujoco/src/render/filament/assets/decor_fade.filamat',
            '/third_party/mujoco/src/render/filament/assets/haze.filamat',
            '/third_party/mujoco/src/render/filament/assets/unlit_depth.filamat',
            '/third_party/mujoco/src/render/filament/assets/unlit_segmentation.filamat',
            '/third_party/mujoco/src/render/filament/assets/unlit_ui.filamat',
            '/third_party/mujoco/studio/data/AtkinsonHyperlegibleNext[wght].ttf',
            '/third_party/mujoco/studio/data/AtkinsonHyperlegibleMono-Regular.ttf',
            '/third_party/mujoco/studio/data/fontawesome-webfont.ttf',
          ];

      const dlEl = document.getElementById('loadingDownload');
      const dlFileEl = document.getElementById('loadingDownloadFile');
      if (dlEl) dlEl.style.display = 'block';
      let loadedAssets = 0;

      const assetPromises = assetsToPrefetch.map(async (relativePath) => {
        const filename =
            relativePath.substring(relativePath.lastIndexOf('/') + 1);
        if (window.__MUJOCO_OVERVIEW_ENGINE_ASSETS__ &&
            window.__MUJOCO_OVERVIEW_ENGINE_ASSETS__[filename] !== undefined) {
          const bytes = decodeBase64Bytes(
              window.__MUJOCO_OVERVIEW_ENGINE_ASSETS__[filename]);
          delete window.__MUJOCO_OVERVIEW_ENGINE_ASSETS__[filename];
          moduleConfig.registerAsset(filename, bytes);
          loadedAssets++;
          if (dlFileEl) {
            dlFileEl.textContent =
                `${loadedAssets} / ${assetsToPrefetch.length} engine assets`;
          }
          return;
        }
        const assetUrl = moduleConfig.locateFile(relativePath);
        const response = await fetch(assetUrl);
        if (!response.ok) {
          throw new Error(
              `Failed to fetch ${assetUrl}: ${response.statusText}`);
        }
        const buffer = await response.arrayBuffer();
        moduleConfig.registerAsset(filename, new Uint8Array(buffer));
        loadedAssets++;
        if (dlFileEl) {
          dlFileEl.textContent =
              `${loadedAssets} / ${assetsToPrefetch.length} engine assets`;
        }
      });

      Promise.all(assetPromises)
          .then(async () => {
            delete window.__MUJOCO_OVERVIEW_ENGINE_ASSETS__;
            try {
              const params = new URLSearchParams(window.location.search);
              const baseURL = getBaseUrl();
              const defaultModelBase = baseURL + '/model';
              const modelBaseUrl = params.get('model_base') || defaultModelBase;
              const themeParam = (params.get('theme') || 'dark').toLowerCase();
              const darkTheme = themeParam !== 'light';
              const orbit = parseFloat(params.get('orbit') || '0') || 0.0;

              moduleConfig.init(modelBaseUrl, darkTheme, orbit);

              const onProgress = (s) => {
                if (dlFileEl) {
                  dlFileEl.textContent = `${s.files} scene files · ${
                      (s.bytes / 1024 / 1024).toFixed(1)} MB`;
                }
              };

              const allPrimed =
                  Boolean(primeAllBundledModels(modelBaseUrl, onProgress));

              if (!allPrimed) {
                const sceneUrl = moduleConfig.getSceneModelUrl();
                const policyUrl = moduleConfig.getPolicyUrl();
                await Promise.all([
                  prefetchModelAssets(sceneUrl, onProgress),
                  prefetchBinaryAsset(policyUrl, onProgress),
                ]);
              }
              await moduleConfig.loadScene();
            } catch (error) {
              console.error('Failed to initialize MuJoCo Homepage:', error);
            } finally {
              if (dlEl) dlEl.style.display = 'none';
              hideLoading();
              requestAnimationFrame(moduleConfig.animate);
            }
          })
          .catch((error) => {
            console.error('Failed to prefetch engine assets:', error);
            hideLoading();
          });
    },
    animate: async () => {
      try {
        await moduleConfig.renderFrame();
      } catch (error) {
        console.error('Update error:', error);
      }
      requestAnimationFrame(moduleConfig.animate);
    },
  };

  window.Module = moduleConfig;

  let lastCanvasW = 0;
  let lastCanvasH = 0;
  /**
   * Resizes the canvas to match the visible viewport in whole CSS pixels.
   * @param {!Event=} e
   */
  function fitCanvas(e) {
    const vv = window.visualViewport;
    const w = Math.max(1, Math.floor(vv ? vv.width : window.innerWidth));
    const h = Math.max(1, Math.floor(vv ? vv.height : window.innerHeight));
    if (vv && (vv.offsetLeft !== 0 || vv.offsetTop !== 0)) {
      window.scrollTo(0, 0);
    }
    moduleConfig.canvas.style.transform =
        vv && (vv.offsetLeft || vv.offsetTop) ?
        `translate(${Math.round(vv.offsetLeft)}px, ${
            Math.round(vv.offsetTop)}px)` :
        '';
    if (w !== lastCanvasW || h !== lastCanvasH) {
      lastCanvasW = w;
      lastCanvasH = h;
      moduleConfig.canvas.style.width = w + 'px';
      moduleConfig.canvas.style.height = h + 'px';
      if (e && e.target !== window) {
        window.dispatchEvent(new Event('resize'));
      }
    }
  }
  fitCanvas();
  window.addEventListener('resize', fitCanvas);
  window.visualViewport?.addEventListener('resize', fitCanvas);
  window.visualViewport?.addEventListener('scroll', fitCanvas);

  const activeTouches = new Map();
  /**
   * Tracks currently active Touch objects on the canvas.
   * @param {!TouchEvent} e
   */
  function syncActiveTouches(e) {
    const remaining = new Set();
    for (const t of e.touches) {
      remaining.add(t.identifier);
      activeTouches.set(t.identifier, t);
    }
    for (const id of activeTouches.keys()) {
      if (!remaining.has(id)) {
        activeTouches.delete(id);
      }
    }
  }
  /**
   * Dispatches a synthetic touchcancel event for any still-active touches.
   */
  function cancelActiveTouches() {
    if (activeTouches.size === 0) return;
    const list = [...activeTouches.values()];
    activeTouches.clear();
    let ev = null;
    if (typeof TouchEvent === 'function') {
      try {
        const candidate = new TouchEvent('touchcancel', {
          bubbles: true,
          cancelable: false,
          touches: [],
          targetTouches: [],
          changedTouches: list,
        });
        if (candidate.changedTouches &&
            candidate.changedTouches.length === list.length) {
          ev = candidate;
        }
      } catch (unused) {
      }
    }
    if (!ev) {
      ev = new Event('touchcancel', {bubbles: true, cancelable: false});
      ev.touches = [];
      ev.targetTouches = [];
      ev.changedTouches = list;
    }
    moduleConfig.canvas.dispatchEvent(ev);
  }
  for (const type of ['touchstart', 'touchmove', 'touchend', 'touchcancel']) {
    moduleConfig.canvas.addEventListener(type, syncActiveTouches, {
      capture: true,
      passive: true,
    });
  }
  window.addEventListener('blur', cancelActiveTouches);
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) cancelActiveTouches();
  });
  window.addEventListener('pagehide', cancelActiveTouches);

  /** Appends the Emscripten JS runtime script and hooks optional CSP WASM. */
  function bootstrapWasmRuntime() {
    if (window.__MUJOCO_OVERVIEW_WASM_B64__) {
      moduleConfig.instantiateWasm = (imports, successCallback) => {
        try {
          const bytes = decodeBase64Bytes(window.__MUJOCO_OVERVIEW_WASM_B64__);
          delete window.__MUJOCO_OVERVIEW_WASM_B64__;
          WebAssembly.instantiate(bytes, imports)
              .then((result) => {
                successCallback(result.instance);
              })
              .catch((err) => {
                console.error('WebAssembly.instantiate failed:', err);
              });
        } catch (err) {
          console.error('Failed to decode bundled WASM:', err);
        }
        return {};
      };
    }
    const script = document.createElement('script');
    script.async = true;
    const runtimeJs = window.MUJOCO_OVERVIEW_STATIC_LAYOUT ?
        'mujoco_homepage.js' :
        'playground_emscripten.js';
    script.src = moduleConfig.locateFile(runtimeJs);
    script.onerror = () => {
      hideLoading();
      console.error('Failed to load MuJoCo Homepage runtime:', script.src);
    };
    document.body.appendChild(script);
  }

  bootstrapWasmRuntime();
})();
