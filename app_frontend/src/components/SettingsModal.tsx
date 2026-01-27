import React, { useState } from 'react';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { useTranslation } from '@/i18n';
import { Switch } from '@/components/ui/switch';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import { Separator } from '@/components/ui/separator';
import { useAppState } from '@/state';
import { useDataRobotInfo, useUpdateApiToken } from '@/api/user/hooks';
import { Input } from './ui/input';
import { LanguageSwitcher } from './LanguageSwitcher';
import { Checkbox } from '@/components/ui/checkbox';
import { useTheme } from '@/theme/theme-provider';
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

  const [localCollapsiblePanelDefaultOpen, setLocalCollapsiblePanelDefaultOpen] = useState(
    collapsiblePanelDefaultOpen
  );
  const [localEnableChartGeneration, setLocalEnableChartGeneration] =
    useState(enableChartGeneration);
  const [localEnableBusinessInsights, setLocalEnableBusinessInsights] =
    useState(enableBusinessInsights);
  const [localIncludeCsvBom, setLocalIncludeCsvBom] = useState(includeCsvBom);

  const handleSaveSettings = () => {
    setCollapsiblePanelDefaultOpen(localCollapsiblePanelDefaultOpen);
    setEnableChartGeneration(localEnableChartGeneration);
    setEnableBusinessInsights(localEnableBusinessInsights);
    setIncludeCsvBom(localIncludeCsvBom);
    onOpenChange(false);
  };

  const { theme, setTheme } = useTheme();

  return (
    <Dialog open={isOpen} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[425px]">
        <DialogHeader>
          <DialogTitle className="text-center">{t('Settings')}</DialogTitle>
          <DialogDescription className="text-center">
            {t('Customize your chat experience')}
          </DialogDescription>
        </DialogHeader>
        <div>
          <div className="flex items-center justify-between gap-4 py-2">
            <Label htmlFor="collapsible-default-open" className="cursor-pointer">
              {t('Expand data panels by default')}
            </Label>
            <Checkbox
              id="collapsible-default-open"
              checked={localCollapsiblePanelDefaultOpen}
              onCheckedChange={value => setLocalCollapsiblePanelDefaultOpen(value as boolean)}
            />
          </div>

          <div className="flex items-center justify-between gap-4 py-2">
            <Label htmlFor="enable-chart-generation" className="cursor-pointer">
              {t('Enable chart generation')}
            </Label>
            <Switch
              id="enable-chart-generation"
              checked={localEnableChartGeneration}
              onCheckedChange={e => setLocalEnableChartGeneration(e)}
            />
          </div>
          <div className="flex items-center justify-between gap-4 py-2">
            <Label htmlFor="enable-business-insights" className="cursor-pointer">
              {t('Enable business insights')}
            </Label>
            <Switch
              id="enable-business-insights"
              checked={localEnableBusinessInsights}
              onCheckedChange={e => setLocalEnableBusinessInsights(e)}
            />
          </div>

          <>
            <Separator className="border-t my-2" />
            <div className="my-4 space-y-4 flex justify-between items-center">
              <p className="mn-label">{t('Language')}</p>
              <LanguageSwitcher />
            </div>
            <div className="my-4 space-y-4 flex justify-between items-center">
              <p className="mn-label">{t('Dark Theme')}</p>
              <Switch
                id="theme"
                checked={theme === 'dark'}
                onCheckedChange={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
              />
            </div>
            <div
              className="flex items-center justify-between gap-4 py-2"
              title={t(
                'Include the byte order mark (BOM) in exported CSVs for better compatibility with international characters.'
              )}
            >
              <Label htmlFor="include-csv-bom" className="cursor-pointer">
                {t('Include BOM')}
              </Label>
              <Switch
                id="include-csv-bom"
                checked={localIncludeCsvBom}
                onCheckedChange={e => setLocalIncludeCsvBom(e)}
              />
            </div>
          </>

          <Separator className="border-t my-2" />

          <div className="mt-4 space-y-4">
            <div className="flex justify-between items-center">
              <h3 className="mn-label">{t('DataRobot Connection')}</h3>
              <Button
                variant="ghost"
                size="sm"
                disabled={isRefreshingConnection}
                onClick={async () => {
                  try {
                    setIsRefreshingConnection(true);
                    setRefreshError(null);
                    await refetchDataRobotInfo();
                  } catch (error) {
                    console.error('Failed to refresh connection:', error);
                    setRefreshError(
                      error instanceof Error ? error.message : t('Failed to connect to DataRobot')
                    );
                  } finally {
                    setIsRefreshingConnection(false);
                  }
                }}
              >
                {isRefreshingConnection ? t('Refreshing...') : t('Refresh')}
              </Button>
            </div>
            {isLoadingDataRobotInfo || isRefreshingConnection ? (
              <p className="body">{t('Loading DataRobot info...')}</p>
            ) : dataRobotInfo?.datarobot_account_info ? (
              <div className="space-y-1">
                <p>
                  <span className="mr-1">{t('Connected as:')}</span>
                  <span>{dataRobotInfo.datarobot_account_info.username}</span>
                </p>
                <p>
                  {t('Email')}: {dataRobotInfo.datarobot_account_info.email}
                </p>
                {dataRobotInfo.datarobot_api_token && (
                  <p>
                    <span className="mr-1">{t('API Token:')}</span>
                    <span className="py-0.5 rounded">{dataRobotInfo.datarobot_api_token}</span>
                  </p>
                )}
                {dataRobotInfo.datarobot_api_scoped_token && (
                  <p>
                    <span className="mr-1">{t('Scoped Token:')}</span>
                    <span className="py-0.5 rounded">
                      {dataRobotInfo.datarobot_api_scoped_token}
                    </span>
                  </p>
                )}
                <p>
                  <a href={`/account/developer-tools`} target="_blank" rel="noopener noreferrer">
                    {t('Manage API keys →')}
                  </a>
                </p>
              </div>
            ) : refreshError ? (
              <div className="space-y-2">
                <p className="text-destructive">{t('Connection error')}</p>
                <p>{refreshError}</p>
                <p>{t('Check your DataRobot connection and try again')}</p>
              </div>
            ) : (
              <div className="space-y-2">
                <p>{t('Not connected to DataRobot')}</p>
                <p>{t('Use the Refresh button to connect if DataRobot is available')}</p>
              </div>
            )}

            <div className="pt-4 border-t">
              <h4 className="font-medium mb-4">{t('Update API Token')}</h4>
              <div className="flex flex-col gap-2">
                <Input
                  type="password"
                  autoComplete="off"
                  value={apiToken}
                  onChange={e => {
                    setApiToken(e.target.value);
                    if (updateApiTokenMutation.isError) {
                      updateApiTokenMutation.reset();
                    }
                  }}
                  placeholder={t('Enter DataRobot API token')}
                  disabled={updateApiTokenMutation.isPending}
                />
                <div className="flex justify-end">
                  <Button
                    variant="ghost"
                    size="sm"
                    disabled={updateApiTokenMutation.isPending || !apiToken.trim()}
                    onClick={() => {
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
                    }}
                  >
                    {updateApiTokenMutation.isPending ? t('Updating...') : t('Update Token')}
                  </Button>
                </div>

                {updateApiTokenMutation.isError && (
                  <p className="text-destructive text-sm mt-1">
                    {updateApiTokenMutation.error instanceof Error
                      ? updateApiTokenMutation.error.message
                      : t('Failed to update API token')}
                  </p>
                )}

                {tokenUpdateSuccess && (
                  <p className="text-success text-sm mt-1">
                    {t('API token updated successfully!')}
                  </p>
                )}

                <p className="body-secondary mt-1">
                  {t('Manually enter your DataRobot API token to authenticate with the service.')}
                </p>
              </div>
            </div>
          </div>
        </div>
        <Separator className="border-t mt-2" />
        <DialogFooter className="mt-4">
          <Button variant="secondary" onClick={() => onOpenChange(false)}>
            {t('Cancel')}
          </Button>
          <Button onClick={handleSaveSettings}>{t('Save changes')}</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};
