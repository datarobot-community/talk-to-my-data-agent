/// <reference types="vitest/globals" />
import '@testing-library/jest-dom';
import { configure } from '@testing-library/react';
import { server } from './__mocks__/node.js';

configure({ asyncUtilTimeout: 5000 });

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
