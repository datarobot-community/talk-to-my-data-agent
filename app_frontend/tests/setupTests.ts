import '@testing-library/jest-dom';
import { server } from './__mocks__/node.js';

// jsdom doesn't implement ResizeObserver (used by the custom Tabs component)
global.ResizeObserver = class ResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
};

beforeAll(() => {
  server.listen();
});

afterEach(() => {
  server.resetHandlers();
});

afterAll(() => {
  server.close();
});
