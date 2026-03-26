import { screen, fireEvent, waitFor, act } from '@testing-library/react';
import { describe, expect, vi, beforeEach, it, type Mock } from 'vitest';
import { SettingsModal } from '@/components/SettingsModal';
import { renderWithProviders } from '../test-utils';
import { useDataRobotInfo, useUpdateApiToken } from '@/api/user/hooks';
import { useTheme } from '@/theme/theme-provider';

vi.mock('@/api/user/hooks', () => ({
  useDataRobotInfo: vi.fn(),
  useUpdateApiToken: vi.fn(),
}));

vi.mock('@/theme/theme-provider', () => ({
  useTheme: vi.fn(),
}));

const mockSetTheme = vi.fn();
const mockRefetch = vi.fn();
const mockMutate = vi.fn();
const mockReset = vi.fn();

const defaultDataRobotInfo = {
  datarobot_account_info: {
    firstName: 'Anton',
    lastName: 'Berezhinsky',
    email: 'anton@datarobot.com',
  },
  datarobot_api_token: '****AbC1',
  datarobot_api_scoped_token: null,
};

function setupMocks(overrides?: {
  dataRobotInfo?: typeof defaultDataRobotInfo | null;
  isLoading?: boolean;
  theme?: 'light' | 'dark';
  mutationState?: { isPending?: boolean; isError?: boolean; error?: Error };
}) {
  (useTheme as Mock).mockReturnValue({
    theme: overrides?.theme ?? 'dark',
    setTheme: mockSetTheme,
  });

  (useDataRobotInfo as Mock).mockReturnValue({
    data: overrides?.dataRobotInfo === undefined ? defaultDataRobotInfo : overrides.dataRobotInfo,
    isLoading: overrides?.isLoading ?? false,
    refetch: mockRefetch,
  });

  (useUpdateApiToken as Mock).mockReturnValue({
    mutate: mockMutate,
    isPending: overrides?.mutationState?.isPending ?? false,
    isError: overrides?.mutationState?.isError ?? false,
    error: overrides?.mutationState?.error ?? null,
    reset: mockReset,
  });
}

describe('SettingsModal', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    setupMocks();
  });

  it('renders all sections with correct structure', () => {
    renderWithProviders(<SettingsModal isOpen={true} onOpenChange={vi.fn()} />);

    expect(screen.getByRole('dialog')).toBeInTheDocument();
    expect(screen.getByLabelText('Dark theme')).toBeInTheDocument();
    expect(screen.getByLabelText('Enable chart generation')).toBeInTheDocument();
    expect(screen.getByLabelText('Enable business insights')).toBeInTheDocument();
    expect(screen.getByLabelText('Include BOM in CSV exports')).toBeInTheDocument();
    expect(screen.getByLabelText('Expand data panels by default')).toBeInTheDocument();
  });

  it('toggles dark theme via setTheme', () => {
    setupMocks({ theme: 'dark' });
    renderWithProviders(<SettingsModal isOpen={true} onOpenChange={vi.fn()} />);

    const themeSwitch = screen.getByLabelText('Dark theme');
    expect(themeSwitch).toBeChecked();

    fireEvent.click(themeSwitch);
    expect(mockSetTheme).toHaveBeenCalledWith('light');
  });

  it('displays connection info with name', () => {
    renderWithProviders(<SettingsModal isOpen={true} onOpenChange={vi.fn()} />);

    expect(screen.getByTestId('connected-as-value')).toHaveTextContent('Anton Berezhinsky');
    expect(screen.getByTestId('email-value')).toHaveTextContent('anton@datarobot.com');
    expect(screen.getByTestId('api-key-value')).toHaveTextContent('****AbC1');
  });

  it('hides Connected as row when name is not provided', () => {
    setupMocks({
      dataRobotInfo: {
        ...defaultDataRobotInfo,
        datarobot_account_info: {
          firstName: '',
          lastName: '',
          email: 'no-name@datarobot.com',
        },
      },
    });
    renderWithProviders(<SettingsModal isOpen={true} onOpenChange={vi.fn()} />);

    expect(screen.queryByTestId('connected-as-value')).not.toBeInTheDocument();
    expect(screen.getByTestId('email-value')).toHaveTextContent('no-name@datarobot.com');
  });

  it('shows own API key in connection grid when scoped token exists', () => {
    setupMocks({
      dataRobotInfo: {
        ...defaultDataRobotInfo,
        datarobot_api_scoped_token: '****XyZ9',
      },
    });
    renderWithProviders(<SettingsModal isOpen={true} onOpenChange={vi.fn()} />);

    expect(screen.getByTestId('own-api-key-value')).toHaveTextContent('****XyZ9');
  });

  it('shows loading state while fetching DataRobot info', () => {
    setupMocks({ dataRobotInfo: null, isLoading: true });
    renderWithProviders(<SettingsModal isOpen={true} onOpenChange={vi.fn()} />);

    expect(screen.getByText('Loading DataRobot info...')).toBeInTheDocument();
    expect(screen.queryByTestId('disconnected-state')).not.toBeInTheDocument();
  });

  it('shows disconnected state when no account info', () => {
    setupMocks({ dataRobotInfo: null });
    renderWithProviders(<SettingsModal isOpen={true} onOpenChange={vi.fn()} />);

    expect(screen.getByTestId('disconnected-state')).toBeInTheDocument();
  });

  it('calls refetch when Refresh is clicked', async () => {
    mockRefetch.mockResolvedValue({});
    renderWithProviders(<SettingsModal isOpen={true} onOpenChange={vi.fn()} />);

    fireEvent.click(screen.getByTestId('refresh-connection-button'));
    await waitFor(() => expect(mockRefetch).toHaveBeenCalled());
  });

  it('enables Update button only when API key is entered', () => {
    renderWithProviders(<SettingsModal isOpen={true} onOpenChange={vi.fn()} />);

    expect(screen.getByTestId('update-api-key-button')).toBeDisabled();

    fireEvent.change(screen.getByPlaceholderText('Enter API key'), {
      target: { value: 'my-new-key' },
    });
    expect(screen.getByTestId('update-api-key-button')).not.toBeDisabled();
  });

  it('calls mutate with entered API key on Update click', () => {
    renderWithProviders(<SettingsModal isOpen={true} onOpenChange={vi.fn()} />);

    fireEvent.change(screen.getByPlaceholderText('Enter API key'), {
      target: { value: 'my-new-key' },
    });
    fireEvent.click(screen.getByTestId('update-api-key-button'));

    expect(mockMutate).toHaveBeenCalledWith('my-new-key', expect.any(Object));

    const onSuccess = mockMutate.mock.calls[0][1].onSuccess;
    act(() => onSuccess());
    expect(screen.getByPlaceholderText('Enter API key')).toHaveValue('');
  });

  it('shows error message when API key update fails', () => {
    setupMocks({
      mutationState: { isError: true, error: new Error('Invalid key') },
    });
    renderWithProviders(<SettingsModal isOpen={true} onOpenChange={vi.fn()} />);

    expect(screen.getByTestId('api-key-error')).toHaveTextContent('Invalid key');
  });

  it('renders Manage API keys link pointing to developer tools', () => {
    renderWithProviders(<SettingsModal isOpen={true} onOpenChange={vi.fn()} />);

    const link = screen.getByTestId('manage-api-keys-link');
    expect(link).toHaveAttribute('href', '/account/developer-tools');
    expect(link).toHaveAttribute('target', '_blank');
  });

  it('shows app version when APP_VERSION is set', () => {
    window.ENV = { APP_VERSION: 'v11.5.2-3-gabcdef0' };
    renderWithProviders(<SettingsModal isOpen={true} onOpenChange={vi.fn()} />);

    expect(screen.getByText(/v11\.5\.2-3-gabcdef0/)).toBeInTheDocument();
  });

  it('hides version section when APP_VERSION is not set', () => {
    window.ENV = {};
    renderWithProviders(<SettingsModal isOpen={true} onOpenChange={vi.fn()} />);

    expect(screen.queryByText(/Version/)).not.toBeInTheDocument();
  });
});
