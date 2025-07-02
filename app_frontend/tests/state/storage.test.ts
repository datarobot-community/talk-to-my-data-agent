import { test, describe, expect, vi } from 'vitest';
import {
  getStorageItem,
  setStorageItem,
  isWelcomeModalHidden,
} from "@/state/storage";
import { STORAGE_KEYS } from "@/state/constants";

describe("Storage Module", () => {
  // Mock localStorage
  const localStorageMock = (() => {
    let store: Record<string, string> = {};
    return {
      getItem: vi.fn((key: string) => store[key] || null),
      setItem: vi.fn((key: string, value: string) => {
        store[key] = value;
      }),
      clear: () => {
        store = {};
      },
    };
  })();

  // Save original window.location
  const originalLocation = window.location;

  beforeEach(() => {
    // Setup localStorage mock
    Object.defineProperty(window, "localStorage", { value: localStorageMock });
    localStorageMock.clear();
    vi.clearAllMocks();

    // Reset mocks
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    delete (window as any).location;
    window.location = {
      ...originalLocation,
      href: "https://app.datarobot.com",
    };
  });

  afterEach(() => {
    // Restore window.location
    window.location = originalLocation;
  });

  describe("getStorageItem", () => {
    test("returns null when item does not exist", () => {
      expect(getStorageItem("non-existent-key")).toBeNull();
      expect(localStorageMock.getItem).toHaveBeenCalledWith("non-existent-key");
    });

    test("returns item value when it exists", () => {
      localStorageMock.getItem.mockReturnValueOnce("test-value");
      expect(getStorageItem("existing-key")).toBe("test-value");
      expect(localStorageMock.getItem).toHaveBeenCalledWith("existing-key");
    });

    test("uses prefixed key when app ID is in URL", () => {
      // Set URL with app ID
      window.location.href =
        "https://app.datarobot.com/custom_applications/67c5a843f01cb4b31c2be037/data";

      getStorageItem("test-key");
      expect(localStorageMock.getItem).toHaveBeenCalledWith(
        "/custom_applications/67c5a843f01cb4b31c2be037/test-key"
      );
    });

    test("uses original key when app ID is not in URL", () => {
      // Set URL without app ID
      window.location.href = "https://app.datarobot.com/some/other/path";

      getStorageItem("test-key");
      expect(localStorageMock.getItem).toHaveBeenCalledWith("test-key");
    });
  });

  describe("setStorageItem", () => {
    test("sets item with original key when app ID is not in URL", () => {
      // Set URL without app ID
      window.location.href = "https://app.datarobot.com/some/other/path";

      setStorageItem("test-key", "test-value");
      expect(localStorageMock.setItem).toHaveBeenCalledWith(
        "test-key",
        "test-value"
      );
    });

    test("sets item with prefixed key when app ID is in URL", () => {
      // Set URL with app ID
      window.location.href =
        "https://app.datarobot.com/custom_applications/67c5a843f01cb4b31c2be037/chats";

      setStorageItem("test-key", "test-value");
      expect(localStorageMock.setItem).toHaveBeenCalledWith(
        "/custom_applications/67c5a843f01cb4b31c2be037/test-key",
        "test-value"
      );
    });

    test("sets item with prefixed key for nested URL paths", () => {
      // Set URL with app ID and nested path
      window.location.href =
        "https://app.datarobot.com/custom_applications/67c5a843f01cb4b31c2be037/chats/f64087df-d5d7-4e86-b4c3-5dcc6f851675";

      setStorageItem("test-key", "test-value");
      expect(localStorageMock.setItem).toHaveBeenCalledWith(
        "/custom_applications/67c5a843f01cb4b31c2be037/test-key",
        "test-value"
      );
    });
  });

  describe("isWelcomeModalHidden", () => {
    test("returns false when HIDE_WELCOME_MODAL is not set", () => {
      expect(isWelcomeModalHidden()).toBe(false);
      expect(localStorageMock.getItem).toHaveBeenCalledWith(
        STORAGE_KEYS.HIDE_WELCOME_MODAL
      );
    });

    test('returns true when HIDE_WELCOME_MODAL is set to "true"', () => {
      // Set URL with app ID
      window.location.href =
        "https://app.datarobot.com/custom_applications/67c5a843f01cb4b31c2be037/data";

      localStorageMock.getItem.mockReturnValueOnce("true");
      expect(isWelcomeModalHidden()).toBe(true);
      expect(localStorageMock.getItem).toHaveBeenCalledWith(
        "/custom_applications/67c5a843f01cb4b31c2be037/HIDE_WELCOME_MODAL"
      );
    });

    test('returns false when HIDE_WELCOME_MODAL is set to something other than "true"', () => {
      localStorageMock.getItem.mockReturnValueOnce("false");
      expect(isWelcomeModalHidden()).toBe(false);
      expect(localStorageMock.getItem).toHaveBeenCalledWith(
        STORAGE_KEYS.HIDE_WELCOME_MODAL
      );
    });
  });
});
