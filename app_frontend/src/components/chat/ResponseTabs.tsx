import React from "react";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { LoadingIndicator } from "./LoadingIndicator";
import { RESPONSE_TABS } from "./constants";

export interface TabState {
  isLoading?: boolean;
  hasError: boolean;
}

interface ResponseTabsProps {
  value: string;
  onValueChange: (value: string) => void;
  tabStates?: {
    summary: TabState;
    insights: TabState;
    code: TabState;
  };
}

export const ResponseTabs: React.FC<ResponseTabsProps> = ({
  value,
  onValueChange,
  tabStates,
}) => {
  const states = tabStates || {
    summary: {
      isLoading: false,
      hasError: false,
    },
    insights: {
      isLoading: false,
      hasError: false,
    },
    code: {
      isLoading: false,
      hasError: false,
    },
  };

  return (
    <Tabs
      defaultValue={RESPONSE_TABS.SUMMARY}
      value={value}
      onValueChange={onValueChange}
      className="w-fit py-4"
    >
      <TabsList className="grid w-full grid-cols-3">
        <TabsTrigger value={RESPONSE_TABS.SUMMARY}>
          <LoadingIndicator
            isLoading={states.summary.isLoading}
            hasError={states.summary.hasError}
            successTestId="summary-loading-success"
          />
          Summary
        </TabsTrigger>
        <TabsTrigger value={RESPONSE_TABS.INSIGHTS}>
          <LoadingIndicator
            isLoading={states.insights.isLoading}
            hasError={states.insights.hasError}
            successTestId="insights-loading-success"
          />
          More insights
        </TabsTrigger>
        <TabsTrigger value={RESPONSE_TABS.CODE}>
          <LoadingIndicator
            isLoading={states.code.isLoading}
            hasError={states.code.hasError}
            successTestId="code-loading-success"
          />
          Behind the scenes
        </TabsTrigger>
      </TabsList>
    </Tabs>
  );
};
