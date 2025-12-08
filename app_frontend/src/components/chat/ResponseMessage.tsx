import React, { useState, useMemo, useRef, useEffect } from 'react';
import {
  IChatMessage,
  IMessageComponent,
  IBusinessComponent,
  IChartsComponent,
  IAnalysisComponent,
  IUsageInfoComponent,
} from '@/api/chat-messages/types';
import { getTranslatedMessageStep } from '@/api/chat-messages/utils';
import { MessageContainer } from './MessageContainer';
import { MessageHeader } from './MessageHeader';

import { ResponseTabs } from './ResponseTabs';
import { SummaryTabContent } from './SummaryTabContent';
import { InsightsTabContent } from './InsightsTabContent';
import { CodeTabContent } from './CodeTabContent';
import { ErrorPanel } from './ErrorPanel';
import { RESPONSE_TABS } from './constants';
import { LoadingIndicator } from './LoadingIndicator';

interface ResponseMessageProps {
  chatId: string;
  message: IChatMessage;
  messages: IChatMessage[];
  hasInProgressMessages: boolean;
  testId?: string;
}

const isMessageComponent = (component: unknown): component is IMessageComponent => {
  return !!component && typeof component === 'object' && 'enhanced_user_message' in component;
};

const isBusinessComponent = (component: unknown): component is IBusinessComponent => {
  return (
    !!component &&
    typeof component === 'object' &&
    'type' in component &&
    (component as { type?: string }).type === 'business'
  );
};

const isChartsComponent = (component: unknown): component is IChartsComponent => {
  return (
    !!component &&
    typeof component === 'object' &&
    'type' in component &&
    (component as { type?: string }).type === 'charts'
  );
};

const isAnalysisComponent = (component: unknown): component is IAnalysisComponent => {
  return (
    !!component &&
    typeof component === 'object' &&
    'type' in component &&
    (component as { type?: string }).type === 'analysis'
  );
};

const isUsageInfoComponent = (component: unknown): component is IUsageInfoComponent => {
  return (
    !!component &&
    typeof component === 'object' &&
    'type' in component &&
    (component as { type?: string }).type === 'usage_info'
  );
};

export const ResponseMessage: React.FC<ResponseMessageProps> = ({
  message,
  chatId,
  messages,
  hasInProgressMessages,
  testId,
}) => {
  const [activeTab, setActiveTab] = useState(RESPONSE_TABS.SUMMARY);
  const isLoading = !!message.in_progress;

  const {
    enhancedUserMessage,
    bottomLine,
    additionalInsights,
    followUpQuestions,
    fig1_json,
    fig2_json,
    code,
    datasetId,
    tabStates,
    analysisErrors,
    chartsErrors,
    businessErrors,
    analysisAttempts,
    messageStep,
  } = useMemo(() => {
    const messageComponent = message?.components?.find(isMessageComponent);
    const businessComponent = message?.components?.find(isBusinessComponent);
    const chartsComponent = message?.components?.find(isChartsComponent);
    const analysisComponent = message?.components?.find(isAnalysisComponent);
    const usageInfoComponent = message?.components?.find(isUsageInfoComponent);

    const enhancedUserMessage = messageComponent?.enhanced_user_message || '';
    const bottomLine = businessComponent?.bottom_line || '';
    const additionalInsights = businessComponent?.additional_insights;
    const followUpQuestions = businessComponent?.follow_up_questions;
    const fig1_json = chartsComponent?.fig1_json || '';
    const fig2_json = chartsComponent?.fig2_json || '';
    const code = analysisComponent?.code || chartsComponent?.code;
    const datasetId = analysisComponent?.dataset_id || null;

    const messageStep = getTranslatedMessageStep(message);

    // Extract errors from components
    const hasBusinessError = businessComponent?.status === 'error';
    const hasAnalysisError = analysisComponent?.status === 'error';
    const hasChartsError = chartsComponent?.status === 'error';

    // Extract error details from metadata
    const analysisErrors =
      hasAnalysisError && analysisComponent?.metadata?.exception?.exception_history;
    const analysisAttempts = analysisComponent?.metadata?.attempts;
    const chartsErrors = hasChartsError && chartsComponent?.metadata?.exception?.exception_history;
    const businessErrors =
      hasBusinessError && businessComponent?.metadata?.exception?.exception_history;

    // Calculate tab states (loading and error indicators)
    const tabStates = {
      summary: {
        isLoading: message?.in_progress && !businessComponent && !hasBusinessError,
        hasError: hasBusinessError,
      },
      insights: {
        isLoading:
          message?.in_progress &&
          (!businessComponent || (!additionalInsights && !hasBusinessError)),
        hasError: hasBusinessError,
      },
      code: {
        isLoading:
          message?.in_progress &&
          !analysisComponent &&
          !chartsComponent?.code &&
          !hasAnalysisError &&
          !hasChartsError,
        hasError: hasAnalysisError || hasChartsError,
      },
    };

    return {
      enhancedUserMessage,
      bottomLine,
      additionalInsights,
      followUpQuestions,
      fig1_json,
      fig2_json,
      code,
      datasetId,
      tabStates,
      analysisErrors,
      chartsErrors,
      businessErrors,
      analysisAttempts,
      usageInfoComponent,
      messageStep,
    };
  }, [message]);

  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    // Scroll to loading and to the summary bottom line when ready
    ref.current?.scrollIntoView({ behavior: 'smooth' });
  }, [isLoading, bottomLine]);

  return (
    <MessageContainer testId={testId} ref={ref} key={message.id}>
      <MessageHeader messageId={message.id} chatId={chatId} messages={messages} />
      <div className="self-stretch text-sm font-normal leading-tight min-w-0">
        {enhancedUserMessage && <div className="mb-3">{enhancedUserMessage}</div>}

        {messageStep && (
          <div className="flex w-full justify-start items-center">
            <LoadingIndicator
              isLoading={true}
              hasError={false}
              successTestId="step-loading-success"
            />
            <div className="ml-1 mb-0 italic text-muted-foreground">{messageStep}</div>
          </div>
        )}

        {message?.error && (
          <div className="max-h-[300px] overflow-x-auto overflow-y-auto max-w-full">
            <span className="text-destructive text-sm">{message?.error}</span>
          </div>
        )}

        <div className="flex items-center justify-between">
          <ResponseTabs value={activeTab} onValueChange={setActiveTab} tabStates={tabStates} />
        </div>

        {activeTab === RESPONSE_TABS.SUMMARY && (
          <>
            {chartsErrors && !analysisErrors && (
              <ErrorPanel errors={chartsErrors} componentType="Charts" />
            )}
            <SummaryTabContent bottomLine={bottomLine} fig1={fig1_json} fig2={fig2_json} />
          </>
        )}

        {activeTab === RESPONSE_TABS.INSIGHTS && (
          <>
            {businessErrors && (
              <ErrorPanel errors={businessErrors} componentType="Business Insights" />
            )}
            <InsightsTabContent
              additionalInsights={additionalInsights}
              followUpQuestions={followUpQuestions}
              chatId={chatId}
              hasInProgressMessages={hasInProgressMessages}
            />
          </>
        )}

        {activeTab === RESPONSE_TABS.CODE && (
          <>
            {analysisErrors && (
              <ErrorPanel
                attempts={analysisAttempts}
                errors={analysisErrors}
                componentType="Analysis"
              />
            )}
            <CodeTabContent code={code} datasetId={datasetId} />
          </>
        )}
      </div>
    </MessageContainer>
  );
};
