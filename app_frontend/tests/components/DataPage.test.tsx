import { screen } from '@testing-library/react';
import { describe, test, expect, vi, beforeEach, afterEach, type Mock } from 'vitest';
import { Data } from '@/pages/Data';
import { renderWithProviders, mockScrollIntoView } from '../test-utils';
import { useGeneratedDictionaries } from '@/api/dictionaries/hooks';
import { useParams } from 'react-router';
import { useNavigate } from 'react-router-dom';

vi.mock('@/api/dictionaries/hooks', () => ({
  useGeneratedDictionaries: vi.fn(),
}));

vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual('react-router-dom');
  return { ...actual, useParams: vi.fn(), useNavigate: vi.fn() };
});

vi.mock('react-router', async () => {
  const actual = await vi.importActual('react-router');
  return {
    ...actual,
    useParams: vi.fn(() => ({ dataId: 'dataset1' })),
    useNavigate: vi.fn(),
    useLocation: vi.fn(() => ({ state: null, pathname: '/data/dataset1' })),
  };
});

vi.mock('@/components/data', async () => {
  const actual = await vi.importActual('@/components/data');
  return {
    ...actual,
    DatasetCardDescriptionPanel: vi.fn(({ dictionary }: any) => (
      <div data-testid={`dataset-card-${dictionary.name}`}>{dictionary.name}</div>
    )),
    ClearDatasetsButton: vi.fn(() => <button data-testid="clear-datasets">Clear</button>),
  };
});

const mockNavigate = vi.fn();

function setupMocks(overrides?: {
  data?: Array<{ name: string; in_progress: boolean; column_descriptions: unknown[] }>;
  status?: string;
  dataId?: string;
}) {
  (useGeneratedDictionaries as Mock).mockReturnValue({
    data: overrides?.data ?? [{ name: 'dataset1', in_progress: false, column_descriptions: [] }],
    status: overrides?.status ?? 'success',
  });
  if (overrides && 'dataId' in overrides) {
    (useParams as Mock).mockReturnValue({ dataId: overrides.dataId });
  }
}

describe('Data Page', () => {
  let cleanupScrollMock: () => void;

  beforeEach(() => {
    vi.clearAllMocks();
    cleanupScrollMock = mockScrollIntoView();
    (useNavigate as Mock).mockReturnValue(mockNavigate);
    setupMocks();
  });

  afterEach(() => {
    cleanupScrollMock();
  });

  test('renders heading and DataViewTabs', () => {
    renderWithProviders(<Data />);
    expect(screen.getByText('Data')).toBeInTheDocument();
    expect(screen.getByTestId('data-view-tabs')).toBeInTheDocument();
  });

  test('shows loading state when status is pending', () => {
    setupMocks({ data: undefined as any, status: 'pending' });
    renderWithProviders(<Data />);
    expect(screen.getByTestId('loading')).toBeInTheDocument();
  });

  test('renders DatasetCardDescriptionPanel for each dictionary', () => {
    setupMocks({
      data: [
        { name: 'dataset1', in_progress: false, column_descriptions: [] },
        { name: 'dataset2', in_progress: false, column_descriptions: [] },
      ],
    });
    renderWithProviders(<Data />);
    expect(screen.getByTestId('dataset-card-dataset1')).toBeInTheDocument();
    expect(screen.getByTestId('dataset-card-dataset2')).toBeInTheDocument();
  });

  test('navigates to first dataset when no dataId in URL', () => {
    setupMocks({
      data: [{ name: 'first-dataset', in_progress: false, column_descriptions: [] }],
      dataId: undefined as any,
    });
    renderWithProviders(<Data />);
    expect(mockNavigate).toHaveBeenCalledWith('/data/first-dataset', expect.any(Object));
  });

  test('navigates to base data route when data array is empty', () => {
    setupMocks({ data: [], dataId: 'gone-dataset' });
    renderWithProviders(<Data />);
    expect(mockNavigate).toHaveBeenCalledWith('/data', expect.any(Object));
  });

  test('shows ClearDatasetsButton when data exists', () => {
    renderWithProviders(<Data />);
    expect(screen.getByTestId('clear-datasets')).toBeInTheDocument();
  });

  test('hides ClearDatasetsButton when no data', () => {
    setupMocks({ data: [] });
    renderWithProviders(<Data />);
    expect(screen.queryByTestId('clear-datasets')).not.toBeInTheDocument();
  });
});
