import React, { useState } from "react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Switch } from "@/components/ui/switch";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";
import { useAppState } from "@/state";
import { useDataRobotInfo, useUpdateApiToken } from "@/api-state/user/hooks";
import { fetchAndStoreDataRobotToken } from "@/api-state/user/api-requests";
import { Input } from "./ui/input";

interface SettingsModalProps {
  isOpen: boolean;
  onOpenChange: (open: boolean) => void;
}

export const SettingsModal: React.FC<SettingsModalProps> = ({
  isOpen,
  onOpenChange,
}) => {
  const {
    collapsiblePanelDefaultOpen,
    setCollapsiblePanelDefaultOpen,
    enableChartGeneration,
    setEnableChartGeneration,
    enableBusinessInsights,
    setEnableBusinessInsights,
    expandGraphsInsightsDefaultOpen,
    setExpandGraphsInsightsDefaultOpen,
  } = useAppState();

  const {
    data: dataRobotInfo,
    isLoading: isLoadingDataRobotInfo,
    refetch: refetchDataRobotInfo,
  } = useDataRobotInfo();
  const updateApiTokenMutation = useUpdateApiToken();
  const [isRefreshingConnection, setIsRefreshingConnection] = useState(false);
  const [refreshError, setRefreshError] = useState<string | null>(null);
  const [apiToken, setApiToken] = useState<string>("");
  const [tokenUpdateSuccess, setTokenUpdateSuccess] = useState(false);

  const [
    localCollapsiblePanelDefaultOpen,
    setLocalCollapsiblePanelDefaultOpen,
  ] = useState(collapsiblePanelDefaultOpen);
  const [localEnableChartGeneration, setLocalEnableChartGeneration] = useState(
    enableChartGeneration
  );
  const [localEnableBusinessInsights, setLocalEnableBusinessInsights] =
    useState(enableBusinessInsights);
  const [localExpandGraphsInsightsDefaultOpen, setLocalExpandGraphsInsightsDefaultOpen] = useState(expandGraphsInsightsDefaultOpen);

  const handleSaveSettings = () => {
    setCollapsiblePanelDefaultOpen(localCollapsiblePanelDefaultOpen);
    setEnableChartGeneration(localEnableChartGeneration);
    setEnableBusinessInsights(localEnableBusinessInsights);
    setExpandGraphsInsightsDefaultOpen(localExpandGraphsInsightsDefaultOpen);
    onOpenChange(false);
  };

  return (
    <Dialog open={isOpen} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[425px]">
        <DialogHeader>
          <DialogTitle className="text-center">Settings</DialogTitle>
          <DialogDescription className="text-center">
            Customize your chat experience
          </DialogDescription>
        </DialogHeader>
        <div>
          <div className="flex items-center justify-between gap-4 py-2">
            <Label
              htmlFor="collapsible-default-open"
              className="cursor-pointer"
            >
              Expand data panels by default
            </Label>
            <input
              id="collapsible-default-open"
              type="checkbox"
              className="h-4 w-4"
              checked={localCollapsiblePanelDefaultOpen}
              onChange={(e) =>
                setLocalCollapsiblePanelDefaultOpen(e.target.checked)
              }
            />
          </div>

          <div className="flex items-center justify-between gap-4 py-2">
            <Label
              htmlFor="expand-graphs-insights-default-open"
              className="cursor-pointer"
            >
              Expand graphs and insights by default
            </Label>
            <input
              id="expand-graphs-insights-default-open"
              type="checkbox"
              className="h-4 w-4"
              checked={localExpandGraphsInsightsDefaultOpen}
              onChange={(e) => setLocalExpandGraphsInsightsDefaultOpen(e.target.checked)}
            />
          </div>

          <div className="flex items-center justify-between gap-4 py-2">
            <Label htmlFor="enable-chart-generation" className="cursor-pointer">
              Enable chart generation
            </Label>
            <Switch
              id="enable-chart-generation"
              checked={localEnableChartGeneration}
              onCheckedChange={(e) => setLocalEnableChartGeneration(e)}
            />
          </div>
          <div className="flex items-center justify-between gap-4 py-2">
            <Label
              htmlFor="enable-business-insights"
              className="cursor-pointer"
            >
              Enable business insights
            </Label>
            <Switch
              id="enable-business-insights"
              checked={localEnableBusinessInsights}
              onCheckedChange={(e) => setLocalEnableBusinessInsights(e)}
            />
          </div>

          <Separator className="border-t my-2" />

          <div className="mt-4 space-y-4">
            <div className="flex justify-between items-center">
              <h3 className="font-semibold">DataRobot Connection</h3>
              <Button
                variant="outline"
                size="sm"
                disabled={isRefreshingConnection}
                onClick={async () => {
                  try {
                    setIsRefreshingConnection(true);
                    setRefreshError(null);
                    await fetchAndStoreDataRobotToken();
                    await refetchDataRobotInfo();
                  } catch (error) {
                    console.error("Failed to refresh connection:", error);
                    setRefreshError(
                      error instanceof Error
                        ? error.message
                        : "Failed to connect to DataRobot"
                    );
                  } finally {
                    setIsRefreshingConnection(false);
                  }
                }}
              >
                {isRefreshingConnection ? "Refreshing..." : "Refresh"}
              </Button>
            </div>
            {isLoadingDataRobotInfo || isRefreshingConnection ? (
              <p className="text-sm">Loading DataRobot info...</p>
            ) : dataRobotInfo?.datarobot_account_info ? (
              <div className="space-y-1">
                <p>
                  <span className="mr-1">Connected as:</span>
                  <span>{dataRobotInfo.datarobot_account_info.username}</span>
                </p>
                <p>Email: {dataRobotInfo.datarobot_account_info.email}</p>
                {dataRobotInfo.datarobot_api_token && (
                  <p>
                    <span className="mr-1">API Token:</span>
                    <span className="py-0.5 rounded">
                      {dataRobotInfo.datarobot_api_token}
                    </span>
                  </p>
                )}
                {dataRobotInfo.datarobot_api_skoped_token && (
                  <p>
                    <span className="mr-1">Scoped Token:</span>
                    <span className="py-0.5 rounded">
                      {dataRobotInfo.datarobot_api_skoped_token}
                    </span>
                  </p>
                )}
                <p>
                  <a
                    href={`/account/developer-tools`}
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    Manage API keys →
                  </a>
                </p>
              </div>
            ) : refreshError ? (
              <div className="space-y-2">
                <p className="text-destructive">Connection error</p>
                <p>{refreshError}</p>
                <p>Check your DataRobot connection and try again</p>
              </div>
            ) : (
              <div className="space-y-2">
                <p>Not connected to DataRobot</p>
                <p>
                  Use the Refresh button to connect if DataRobot is available
                </p>
              </div>
            )}

            <div className="pt-4 border-t">
              <h4 className="font-medium mb-4">Update API Token</h4>
              <div className="flex flex-col gap-2">
                <Input
                  type="password"
                  autoComplete="off"
                  value={apiToken}
                  onChange={(e) => {
                    setApiToken(e.target.value);
                    if (updateApiTokenMutation.isError) {
                      updateApiTokenMutation.reset();
                    }
                  }}
                  placeholder="Enter DataRobot API token"
                  disabled={updateApiTokenMutation.isPending}
                />
                <div className="flex justify-end">
                  <Button
                    variant="outline"
                    size="sm"
                    disabled={
                      updateApiTokenMutation.isPending || !apiToken.trim()
                    }
                    onClick={() => {
                      if (!apiToken.trim()) {
                        return;
                      }
                      setTokenUpdateSuccess(false);
                      updateApiTokenMutation.mutate(apiToken, {
                        onSuccess: () => {
                          setApiToken("");
                          setTokenUpdateSuccess(true);
                          setTimeout(() => setTokenUpdateSuccess(false), 3000);
                        },
                      });
                    }}
                  >
                    {updateApiTokenMutation.isPending
                      ? "Updating..."
                      : "Update Token"}
                  </Button>
                </div>

                {updateApiTokenMutation.isError && (
                  <p className="text-destructive text-sm mt-1">
                    {updateApiTokenMutation.error instanceof Error
                      ? updateApiTokenMutation.error.message
                      : "Failed to update API token"}
                  </p>
                )}

                {tokenUpdateSuccess && (
                  <p className="text-success text-sm mt-1">
                    API token updated successfully!
                  </p>
                )}

                <p className="text-xs text-muted-foreground mt-1">
                  Manually enter your DataRobot API token to authenticate with
                  the service.
                </p>
              </div>
            </div>
          </div>
        </div>
        <Separator className="border-t mt-2" />
        <DialogFooter className="mt-4">
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button onClick={handleSaveSettings}>Save changes</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};
