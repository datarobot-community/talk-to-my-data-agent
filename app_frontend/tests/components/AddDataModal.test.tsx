import { useState } from 'react';
import { screen, fireEvent, act } from '@testing-library/react';
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

// AddDataModal is controlled by its parent (the AddDataMenu dropdown). This
// wrapper renders it open so the dialog content is asserted directly, and lets
// the Cancel/close paths flip `open` back to false.
const ControlledAddDataModal = () => {
  const [open, setOpen] = useState(true);
  return <AddDataModal open={open} onOpenChange={setOpen} />;
};

describe('AddDataModal', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    setupMocks();
  });

  test('renders dialog when open', () => {
    renderWithProviders(<ControlledAddDataModal />);
    expect(screen.getByRole('dialog')).toBeInTheDocument();
  });

  test('renders all 4 data source radio options', () => {
    renderWithProviders(<ControlledAddDataModal />);
    expect(screen.getByLabelText('Local file or Data Registry')).toBeInTheDocument();
    expect(screen.getByLabelText('Remote Data Registry')).toBeInTheDocument();
    // Target the radio by role — 'Database' also matches the dialog title now.
    expect(screen.getByRole('radio', { name: 'Database' })).toBeInTheDocument();
    expect(screen.getByLabelText('Remote Data Connections')).toBeInTheDocument();
  });

  test('disables Remote Data Registry when access denied', () => {
    setupMocks({ accessDenied: { datasetRegistry: true } });
    renderWithProviders(<ControlledAddDataModal />);
    expect(screen.getByLabelText('Remote Data Registry')).toBeDisabled();
  });

  test('disables Remote Data Connections when access denied', () => {
    setupMocks({ accessDenied: { dataStore: true } });
    renderWithProviders(<ControlledAddDataModal />);
    expect(screen.getByLabelText('Remote Data Connections')).toBeDisabled();
  });

  // TODO(DM-21387): AddDataModal is temporarily database-only (non-database
  // sections hidden). Restore when the multi-source paths are refactored.
  test.skip('FILE source shows local files section and privacy notice', () => {
    setupMocks({ dataSource: 'file' });
    renderWithProviders(<ControlledAddDataModal />);
    expect(screen.getByText('Local files')).toBeInTheDocument();
    expect(screen.getByText(/Do not upload datasets containing sensitive/)).toBeInTheDocument();
  });

  test.skip('FILE source shows Data Registry when not access denied', () => {
    setupMocks({
      dataSource: 'file',
      datasets: {
        local: [{ id: '1', name: 'dataset1.csv', size: '10MB' }],
        remote: [],
      },
    });
    renderWithProviders(<ControlledAddDataModal />);
    expect(screen.getByText('Data Registry')).toBeInTheDocument();
  });

  test('FILE source hides Data Registry when access denied', () => {
    setupMocks({
      dataSource: 'file',
      accessDenied: { datasetRegistry: true },
    });
    renderWithProviders(<ControlledAddDataModal />);
    expect(screen.queryByText('Data Registry')).not.toBeInTheDocument();
  });

  test('DATABASE source shows database tables section', () => {
    setupMocks({
      dataSource: 'database',
      dbTables: ['table1', 'table2'],
    });
    renderWithProviders(<ControlledAddDataModal />);
    expect(screen.getByText('Select one or more tables')).toBeInTheDocument();
  });

  test.skip('REMOTE_CATALOG source shows remote data registry', () => {
    setupMocks({ dataSource: 'remote_catalog' });
    renderWithProviders(<ControlledAddDataModal />);
    expect(screen.getByText('Data Registry')).toBeInTheDocument();
    expect(screen.getByText('Select one or more catalog items')).toBeInTheDocument();
  });

  test.skip('NEW_DATA_STORE source shows data store selectors', () => {
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
    renderWithProviders(<ControlledAddDataModal />);
    expect(screen.getByText('Add External Data Source')).toBeInTheDocument();
    expect(screen.getByText('Select a data store')).toBeInTheDocument();
    expect(screen.getByText('Select one or more data sources')).toBeInTheDocument();
  });

  test('Save button is disabled with FILE source and no selections', () => {
    setupMocks({ dataSource: 'file' });
    renderWithProviders(<ControlledAddDataModal />);
    expect(screen.getByTestId('add-data-modal-save-button')).toBeDisabled();
  });

  test('Save button is disabled with REMOTE_CATALOG source and no selections', () => {
    setupMocks({ dataSource: 'remote_catalog' });
    renderWithProviders(<ControlledAddDataModal />);
    expect(screen.getByTestId('add-data-modal-save-button')).toBeDisabled();
  });

  test('Save button is disabled with DATABASE source and no tables selected', () => {
    setupMocks({ dataSource: 'database', dbTables: ['t1'] });
    renderWithProviders(<ControlledAddDataModal />);
    expect(screen.getByTestId('add-data-modal-save-button')).toBeDisabled();
  });

  test('Save button is disabled with NEW_DATA_STORE source and no store selected', () => {
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
    renderWithProviders(<ControlledAddDataModal />);
    expect(screen.getByTestId('add-data-modal-save-button')).toBeDisabled();
  });

  test('Cancel button closes dialog', () => {
    renderWithProviders(<ControlledAddDataModal />);
    expect(screen.getByRole('dialog')).toBeInTheDocument();
    fireEvent.click(screen.getByTestId('add-data-modal-cancel-button'));
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
  });

  test.skip('error alert shown on mutation error', () => {
    let capturedOnError: (error: { message: string }) => void;
    (useFileUploadMutation as Mock).mockImplementation(({ onError }: any) => {
      capturedOnError = onError;
      return { mutate: vi.fn(), progress: 0 };
    });
    renderWithProviders(<ControlledAddDataModal />);
    act(() => capturedOnError!({ message: 'Upload failed' }));
    expect(screen.getByText('Upload failed')).toBeInTheDocument();
  });
});
