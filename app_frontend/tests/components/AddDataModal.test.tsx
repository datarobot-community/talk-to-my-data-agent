import { screen, fireEvent } from '@testing-library/react';
import { describe, test, expect, vi, beforeEach, type Mock } from 'vitest';
import { AddDataModal } from '@/components/AddDataModal';
import { renderWithProviders } from '../test-utils';
import {
  useFetchDatasets,
  useFileUploadMutation,
  useGetSupportedDataSourceTypes,
} from '@/api/datasets/hooks';
import { useGetDatabaseTables, useLoadFromDatabaseMutation } from '@/api/database/hooks';
import { useListAvailableDataStores, useSelectDataSourcesMutation } from '@/api/datasources/hooks';
import { useAppState } from '@/state/hooks';

vi.mock('@/api/datasets/hooks', () => ({
  useFetchDatasets: vi.fn(),
  useFileUploadMutation: vi.fn(),
  useGetSupportedDataSourceTypes: vi.fn(),
}));

vi.mock('@/api/database/hooks', () => ({
  useGetDatabaseTables: vi.fn(),
  useLoadFromDatabaseMutation: vi.fn(),
}));

vi.mock('@/api/datasources/hooks', () => ({
  useListAvailableDataStores: vi.fn(),
  useSelectDataSourcesMutation: vi.fn(),
}));

vi.mock('@/state/hooks', () => ({
  useAppState: vi.fn(),
}));

const mockMutate = vi.fn();
const mockLoadFromDatabase = vi.fn();
const mockSelectDataSources = vi.fn();
const mockSetDataSource = vi.fn();

function setupMocks(overrides?: {
  dataSource?: string;
  datasets?: {
    local?: Array<{ id: string; name: string; size: string }>;
    remote?: Array<{ id: string; name: string; size: string }>;
  };
  dbTables?: string[];
  dataStores?: Array<{
    id: string;
    canonical_name: string;
    driver_class_type: string;
    defined_data_sources: Array<{
      data_store_id: string;
      database_catalog: string | null;
      database_schema: string | null;
      database_table: string | null;
    }>;
  }>;
  accessDenied?: { datasetRegistry?: boolean; dataStore?: boolean };
}) {
  const datasetRegistryError = overrides?.accessDenied?.datasetRegistry
    ? { response: { data: { detail: { code: 'USER_ACCESS_DENIED' } } } }
    : null;

  const dataStoreError = overrides?.accessDenied?.dataStore
    ? { response: { data: { detail: { code: 'USER_ACCESS_DENIED' } } } }
    : null;

  (useFetchDatasets as Mock).mockReturnValue({
    data: overrides?.datasets ?? { local: [], remote: [] },
    isLoading: false,
    error: datasetRegistryError,
  });

  (useFileUploadMutation as Mock).mockImplementation(({ onSuccess }: any) => ({
    mutate: mockMutate.mockImplementation(() => onSuccess?.({})),
    progress: 0,
  }));

  (useGetSupportedDataSourceTypes as Mock).mockReturnValue({
    data: null,
    isLoading: false,
  });

  (useGetDatabaseTables as Mock).mockReturnValue({
    data: overrides?.dbTables ?? [],
  });

  (useLoadFromDatabaseMutation as Mock).mockImplementation(({ onSuccess }: any) => ({
    mutate: mockLoadFromDatabase.mockImplementation(() => onSuccess?.({})),
  }));

  (useListAvailableDataStores as Mock).mockReturnValue({
    data: overrides?.dataStores ?? [],
    isLoading: false,
    error: dataStoreError,
  });

  (useSelectDataSourcesMutation as Mock).mockImplementation(({ onSuccess }: any) => ({
    mutate: mockSelectDataSources.mockImplementation(() => onSuccess?.({})),
  }));

  (useAppState as Mock).mockReturnValue({
    dataSource: overrides?.dataSource ?? 'file',
    setDataSource: mockSetDataSource,
  });
}

describe('AddDataModal', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    setupMocks();
  });

  const openModal = () => {
    const button = screen.getByTestId('add-data-button');
    fireEvent.click(button);
  };

  test('renders trigger button', () => {
    renderWithProviders(<AddDataModal />);
    expect(screen.getByTestId('add-data-button')).toBeInTheDocument();
  });

  test('dialog opens on button click', () => {
    renderWithProviders(<AddDataModal />);
    openModal();
    expect(screen.getByRole('dialog')).toBeInTheDocument();
  });

  test('renders all 4 data source radio options', () => {
    renderWithProviders(<AddDataModal />);
    openModal();
    expect(screen.getByLabelText('Local file or Data Registry')).toBeInTheDocument();
    expect(screen.getByLabelText('Remote Data Registry')).toBeInTheDocument();
    expect(screen.getByLabelText('Database')).toBeInTheDocument();
    expect(screen.getByLabelText('Remote Data Connections')).toBeInTheDocument();
  });

  test('disables Remote Data Registry when access denied', () => {
    setupMocks({ accessDenied: { datasetRegistry: true } });
    renderWithProviders(<AddDataModal />);
    openModal();
    expect(screen.getByLabelText('Remote Data Registry')).toBeDisabled();
  });

  test('disables Remote Data Connections when access denied', () => {
    setupMocks({ accessDenied: { dataStore: true } });
    renderWithProviders(<AddDataModal />);
    openModal();
    expect(screen.getByLabelText('Remote Data Connections')).toBeDisabled();
  });

  test('FILE source shows local files section and privacy notice', () => {
    setupMocks({ dataSource: 'file' });
    renderWithProviders(<AddDataModal />);
    openModal();
    expect(screen.getByText('Local files')).toBeInTheDocument();
    expect(screen.getByText(/Do not upload datasets containing sensitive/)).toBeInTheDocument();
  });

  test('FILE source shows Data Registry when not access denied', () => {
    setupMocks({
      dataSource: 'file',
      datasets: {
        local: [{ id: '1', name: 'dataset1.csv', size: '10MB' }],
        remote: [],
      },
    });
    renderWithProviders(<AddDataModal />);
    openModal();
    expect(screen.getByText('Data Registry')).toBeInTheDocument();
  });

  test('FILE source hides Data Registry when access denied', () => {
    setupMocks({
      dataSource: 'file',
      accessDenied: { datasetRegistry: true },
    });
    renderWithProviders(<AddDataModal />);
    openModal();
    expect(screen.queryByText('Data Registry')).not.toBeInTheDocument();
  });

  test('DATABASE source shows database tables section', () => {
    setupMocks({
      dataSource: 'database',
      dbTables: ['table1', 'table2'],
    });
    renderWithProviders(<AddDataModal />);
    openModal();
    expect(screen.getByText('Databases')).toBeInTheDocument();
    expect(screen.getByText('Select one or more tables')).toBeInTheDocument();
  });

  test('REMOTE_CATALOG source shows remote data registry', () => {
    setupMocks({ dataSource: 'remote_catalog' });
    renderWithProviders(<AddDataModal />);
    openModal();
    expect(screen.getByText('Data Registry')).toBeInTheDocument();
    expect(screen.getByText('Select one or more catalog items')).toBeInTheDocument();
  });

  test('NEW_DATA_STORE source shows data store selectors', () => {
    setupMocks({
      dataSource: 'new_data_store',
      dataStores: [
        {
          id: 'ds1',
          canonical_name: 'My Store',
          driver_class_type: 'pg',
          defined_data_sources: [],
        },
      ],
    });
    renderWithProviders(<AddDataModal />);
    openModal();
    expect(screen.getByText('Add External Data Source')).toBeInTheDocument();
    expect(screen.getByText('Select a data store')).toBeInTheDocument();
    expect(screen.getByText('Select one or more data sources')).toBeInTheDocument();
  });

  test('Save with FILE source calls mutate', async () => {
    setupMocks({ dataSource: 'file' });
    renderWithProviders(<AddDataModal />);
    openModal();
    const saveButton = screen.getByTestId('add-data-modal-save-button');
    fireEvent.click(saveButton);
    expect(mockMutate).toHaveBeenCalled();
  });

  test('Save with DATABASE source and no tables selected does not call mutate', async () => {
    setupMocks({ dataSource: 'database', dbTables: ['t1'] });
    renderWithProviders(<AddDataModal />);
    openModal();
    const saveButton = screen.getByTestId('add-data-modal-save-button');
    fireEvent.click(saveButton);
    // No tables selected, so loadFromDatabase should NOT be called
    expect(mockLoadFromDatabase).not.toHaveBeenCalled();
  });

  test('Save with NEW_DATA_STORE source and no store selected does not call mutate', async () => {
    setupMocks({
      dataSource: 'new_data_store',
      dataStores: [
        {
          id: 'ds1',
          canonical_name: 'My Store',
          driver_class_type: 'pg',
          defined_data_sources: [],
        },
      ],
    });
    renderWithProviders(<AddDataModal />);
    openModal();
    const saveButton = screen.getByTestId('add-data-modal-save-button');
    fireEvent.click(saveButton);
    expect(mockSelectDataSources).not.toHaveBeenCalled();
  });

  test('Cancel button closes dialog', () => {
    renderWithProviders(<AddDataModal />);
    openModal();
    expect(screen.getByRole('dialog')).toBeInTheDocument();
    fireEvent.click(screen.getByTestId('add-data-modal-cancel-button'));
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
  });

  test('error alert shown on mutation error', () => {
    (useFileUploadMutation as Mock).mockImplementation(({ onError }: any) => ({
      mutate: vi.fn(() => onError({ message: 'Upload failed' })),
      progress: 0,
    }));
    renderWithProviders(<AddDataModal />);
    openModal();
    fireEvent.click(screen.getByTestId('add-data-modal-save-button'));
    expect(screen.getByText('Upload failed')).toBeInTheDocument();
  });

  test('highlight prop passes through to trigger button', () => {
    renderWithProviders(<AddDataModal highlight />);
    const button = screen.getByTestId('add-data-button');
    expect(button).toHaveAttribute('data-highlight', 'true');
  });
});
