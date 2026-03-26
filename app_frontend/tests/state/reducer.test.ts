import { describe, test, expect, vi, beforeEach } from 'vitest';
import { createInitialState, reducer } from '@/state/reducer';
import { ACTION_TYPES } from '@/state/constants';
import { DATA_SOURCES } from '@/constants/dataSources';

vi.mock('@/state/storage', () => ({
  getStorageItem: vi.fn(() => null),
  setStorageItem: vi.fn(),
}));

import { getStorageItem, setStorageItem } from '@/state/storage';

const mockGetStorageItem = vi.mocked(getStorageItem);
const mockSetStorageItem = vi.mocked(setStorageItem);

beforeEach(() => {
  vi.clearAllMocks();
});

describe('createInitialState', () => {
  test('reads persisted values from storage and inverts boolean flags', () => {
    mockGetStorageItem.mockImplementation(key => {
      const stored: Record<string, string> = {
        HIDE_WELCOME_MODAL: 'true',
        COLLAPSIBLE_PANEL_DEFAULT_OPEN: 'true',
        ENABLE_CHART_GENERATION: 'false',
        ENABLE_BUSINESS_INSIGHTS: 'false',
        INCLUDE_CSV_BOM: 'true',
        DATA_SOURCE: 'database',
      };
      return stored[key] ?? null;
    });
    const state = createInitialState();
    expect(state).toEqual({
      showWelcome: false,
      collapsiblePanelDefaultOpen: true,
      enableChartGeneration: false,
      enableBusinessInsights: false,
      includeCsvBom: true,
      dataSource: 'database',
    });
  });
});

describe('reducer', () => {
  test('persists state change to storage', () => {
    const state = {
      showWelcome: true,
      collapsiblePanelDefaultOpen: false,
      enableChartGeneration: true,
      enableBusinessInsights: true,
      includeCsvBom: false,
      dataSource: DATA_SOURCES.FILE,
    };
    const newState = reducer(state, {
      type: ACTION_TYPES.SET_ENABLE_CHART_GENERATION,
      payload: false,
    });
    expect(newState.enableChartGeneration).toBe(false);
    expect(mockSetStorageItem).toHaveBeenCalledWith('ENABLE_CHART_GENERATION', 'false');
  });
});
