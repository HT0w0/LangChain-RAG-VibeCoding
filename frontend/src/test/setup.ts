import '@testing-library/jest-dom/vitest'

/**
 * jsdom 里没有这两个 API,而 antd 的栅格、Table、Modal 都会调它们。
 *
 * 不打这两个桩的话,任何渲染 antd 组件的测试都会挂在
 * `window.matchMedia is not a function` 上——而那是个**测试环境**的
 * 毛病,看起来却像组件写错了。
 */
if (!window.matchMedia) {
  window.matchMedia = ((query: string) => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: () => {},
    removeListener: () => {},
    addEventListener: () => {},
    removeEventListener: () => {},
    dispatchEvent: () => false,
  })) as unknown as typeof window.matchMedia
}

if (!globalThis.ResizeObserver) {
  globalThis.ResizeObserver = class {
    observe() {}
    unobserve() {}
    disconnect() {}
  } as unknown as typeof ResizeObserver
}
