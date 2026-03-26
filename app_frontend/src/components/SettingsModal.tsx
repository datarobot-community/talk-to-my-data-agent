import React, { useState } from 'react';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { useTranslation } from '@/i18n';
import { Switch } from '@/components/ui/switch';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import { Field, FieldDescription, FieldGroup, FieldSeparator } from '@/components/ui/field';
import { useAppState } from '@/state';
import { useDataRobotInfo, useUpdateApiToken } from '@/api/user/hooks';
import { Input } from './ui/input';
import { LanguageSwitcher } from './LanguageSwitcher';
import { useTheme } from '@/theme/theme-provider';
import { ExternalLink, RefreshCw } from 'lucide-react';

interface SettingsModalProps {
  isOpen: boolean;
  onOpenChange: (open: boolean) => void;
}

export const SettingsModal: React.FC<SettingsModalProps> = ({ isOpen, onOpenChange }) => {
  const { t } = useTranslation();
  const {
    collapsiblePanelDefaultOpen,
    setCollapsiblePanelDefaultOpen,
    enableChartGeneration,
    setEnableChartGeneration,
    enableBusinessInsights,
    setEnableBusinessInsights,
    includeCsvBom,
    setIncludeCsvBom,
  } = useAppState();

  const {
    data: dataRobotInfo,
    isLoading: isLoadingDataRobotInfo,
    refetch: refetchDataRobotInfo,
  } = useDataRobotInfo();
  const updateApiTokenMutation = useUpdateApiToken();
  const [isRefreshingConnection, setIsRefreshingConnection] = useState(false);
  const [refreshError, setRefreshError] = useState<string | null>(null);
  const [apiToken, setApiToken] = useState<string>('');
  const [tokenUpdateSuccess, setTokenUpdateSuccess] = useState(false);

  const { theme, setTheme } = useTheme();

  const { firstName, lastName } = dataRobotInfo?.datarobot_account_info ?? {};
  const fullName = [firstName, lastName].filter(Boolean).join(' ');

  const handleRefreshConnection = async () => {
    try {
      setIsRefreshingConnection(true);
      setRefreshError(null);
      await refetchDataRobotInfo();
    } catch (error) {
      console.error('Failed to refresh connection:', error);
      setRefreshError(error instanceof Error ? error.message : t('Failed to connect to DataRobot'));
    } finally {
      setIsRefreshingConnection(false);
    }
  };

  const handleApiTokenChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setApiToken(e.target.value);
    if (updateApiTokenMutation.isError) {
      updateApiTokenMutation.reset();
    }
  };

  const handleUpdateApiToken = () => {
    if (!apiToken.trim()) {
      return;
    }
    setTokenUpdateSuccess(false);
    updateApiTokenMutation.mutate(apiToken, {
      onSuccess: () => {
        setApiToken('');
        setTokenUpdateSuccess(true);
        setTimeout(() => setTokenUpdateSuccess(false), 3000);
      },
    });
  };

  return (
    <Dialog open={isOpen} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[425px]">
        <DialogHeader>
          <DialogTitle className="text-center">{t('Settings')}</DialogTitle>
          <DialogDescription className="sr-only">
            {t('Configure app preferences and DataRobot connection')}
          </DialogDescription>
        </DialogHeader>

        <FieldGroup>
          {/* General */}
          <p className="uppercased">{t('General')}</p>

          <Field orientation="horizontal" className="gap-2">
            <Switch
              id="theme"
              checked={theme === 'dark'}
              onCheckedChange={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
            />
            <Label htmlFor="theme" className="cursor-pointer">
              {t('Dark theme')}
            </Label>
          </Field>

          <Field orientation="horizontal" className="gap-2">
            <Switch
              id="collapsible-default-open"
              checked={collapsiblePanelDefaultOpen}
              onCheckedChange={value => setCollapsiblePanelDefaultOpen(!!value)}
            />
            <Label htmlFor="collapsible-default-open" className="cursor-pointer">
              {t('Expand data panels by default')}
            </Label>
          </Field>

          <LanguageSwitcher />

          <FieldSeparator />

          {/* Chat */}
          <p className="uppercased">{t('Chat')}</p>

          <Field orientation="horizontal" className="gap-2">
            <Switch
              id="enable-chart-generation"
              checked={enableChartGeneration}
              onCheckedChange={e => setEnableChartGeneration(e)}
            />
            <Label htmlFor="enable-chart-generation" className="cursor-pointer">
              {t('Enable chart generation')}
            </Label>
          </Field>

          <Field orientation="horizontal" className="gap-2">
            <Switch
              id="enable-business-insights"
              checked={enableBusinessInsights}
              onCheckedChange={e => setEnableBusinessInsights(e)}
            />
            <Label htmlFor="enable-business-insights" className="cursor-pointer">
              {t('Enable business insights')}
            </Label>
          </Field>

          <div>
            <Field orientation="horizontal" className="gap-2">
              <Switch
                id="include-csv-bom"
                checked={includeCsvBom}
                onCheckedChange={e => setIncludeCsvBom(e)}
              />
              <Label htmlFor="include-csv-bom" className="cursor-pointer">
                {t('Include BOM in CSV exports')}
              </Label>
            </Field>
            <p className="mt-2 body-secondary">
              {t('Ensures Japanese and Korean characters display correctly')}
            </p>
          </div>

          <FieldSeparator />

          {/* DataRobot Connection */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <p className="uppercased">{t('DataRobot Connection')}</p>
              {isRefreshingConnection && <RefreshCw className="size-3 animate-spin" />}
            </div>
            <Button
              data-testid="refresh-connection-button"
              variant="ghost"
              size="sm"
              disabled={isRefreshingConnection}
              onClick={handleRefreshConnection}
            >
              {t('Refresh')}
            </Button>
          </div>

          {isLoadingDataRobotInfo ? (
            <p className="body">{t('Loading DataRobot info...')}</p>
          ) : dataRobotInfo?.datarobot_account_info ? (
            <div
              data-testid="connection-info"
              className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1"
            >
              {fullName && (
                <>
                  <span className="body-secondary">{t('Connected as')}</span>
                  <span data-testid="connected-as-value">{fullName}</span>
                </>
              )}
              <span className="body-secondary">{t('Email')}</span>
              <span data-testid="email-value">{dataRobotInfo.datarobot_account_info.email}</span>
              {dataRobotInfo.datarobot_api_token && (
                <>
                  <span className="body-secondary">{t('API key')}</span>
                  <span data-testid="api-key-value">{dataRobotInfo.datarobot_api_token}</span>
                </>
              )}
              {dataRobotInfo?.datarobot_api_scoped_token && (
                <>
                  <span className="body-secondary">{t('Own API key')}</span>
                  <span data-testid="own-api-key-value">
                    {dataRobotInfo.datarobot_api_scoped_token}
                  </span>
                </>
              )}
            </div>
          ) : refreshError ? (
            <div data-testid="connection-error" className="space-y-2">
              <p className="text-destructive">{t('Connection error')}</p>
              <p>{refreshError}</p>
              <p>{t('Check your DataRobot connection and try again')}</p>
            </div>
          ) : (
            <div data-testid="disconnected-state" className="space-y-2">
              <p>{t('Not connected to DataRobot')}</p>
              <p>{t('Use the Refresh button to connect if DataRobot is available')}</p>
            </div>
          )}

          <div>
            <FieldDescription className="mb-3">
              {t('Use your own API key instead of the app default')}
            </FieldDescription>

            <div className="flex gap-2">
              <Input
                type="password"
                autoComplete="off"
                value={apiToken}
                onChange={handleApiTokenChange}
                placeholder={t('Enter API key')}
                disabled={updateApiTokenMutation.isPending}
                className="flex-1"
              />
              <Button
                data-testid="update-api-key-button"
                variant="ghost"
                size="sm"
                disabled={updateApiTokenMutation.isPending || !apiToken.trim()}
                onClick={handleUpdateApiToken}
              >
                {updateApiTokenMutation.isPending ? t('Updating...') : t('Update')}
              </Button>
            </div>

            {updateApiTokenMutation.isError && (
              <p data-testid="api-key-error" className="mt-1 text-sm text-destructive">
                {updateApiTokenMutation.error instanceof Error
                  ? updateApiTokenMutation.error.message
                  : t('Failed to update API key')}
              </p>
            )}

            {tokenUpdateSuccess && (
              <p className="mt-1 text-sm text-success">{t('API key updated successfully!')}</p>
            )}
          </div>

          <a
            data-testid="manage-api-keys-link"
            href="/account/developer-tools"
            target="_blank"
            rel="noopener noreferrer"
            className="w-fit anchor"
          >
            {t('Manage API keys')}
            <ExternalLink className="ml-1 inline size-3" />
          </a>

          {window.ENV?.APP_VERSION && (
            <>
              <FieldSeparator />
              <p className="body-secondary text-center">
                {t('Version')}: {window.ENV.APP_VERSION}
              </p>
            </>
          )}
        </FieldGroup>
      </DialogContent>
    </Dialog>
  );
};
