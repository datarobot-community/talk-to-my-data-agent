import React, { useState, useMemo } from 'react';
import {
    IChatMessage,
    IMessageComponent,
    IBusinessComponent,
    IChartsComponent,
    IAnalysisComponent,
} from '@/api/chat-messages/types';
import { MessageContainer } from './MessageContainer';
import { MessageHeader } from './MessageHeader';
import { DataRobotAvatar } from './Avatars';
import { Loading } from './Loading';
import { ResponseTabs } from './ResponseTabs';
import { SummaryTabContent } from './SummaryTabContent';
import { InsightsTabContent } from './InsightsTabContent';
import { CodeTabContent } from './CodeTabContent';
import { ErrorPanel } from './ErrorPanel';
import { RESPONSE_TABS } from './constants';
import { formatMessageDate } from './utils';

interface ResponseMessageProps {
    chatId?: string;
    date?: string;
    message?: IChatMessage;
    isLoading?: boolean;
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

export const ResponseMessage: React.FC<ResponseMessageProps> = ({
    date,
    message,
    chatId,
    isLoading = false,
}) => {
    const [activeTab, setActiveTab] = useState(RESPONSE_TABS.SUMMARY);

    const {
        displayDate,
        enhancedUserMessage,
        bottomLine,
        additionalInsights,
        followUpQuestions,
        fig1_json,
        fig2_json,
        dataset,
        code,
        tabStates,
        analysisErrors,
        chartsErrors,
        businessErrors,
        analysisAttempts,
    } = useMemo(() => {
        const displayDate = message?.created_at
            ? formatMessageDate(message.created_at)
            : date || '';

        const messageComponent = message?.components?.find(isMessageComponent);
        const businessComponent = message?.components?.find(isBusinessComponent);
        const chartsComponent = message?.components?.find(isChartsComponent);
        const analysisComponent = message?.components?.find(isAnalysisComponent);

        const enhancedUserMessage = messageComponent?.enhanced_user_message || '';
        const bottomLine = businessComponent?.bottom_line || '';
        const additionalInsights = businessComponent?.additional_insights;
        const followUpQuestions = businessComponent?.follow_up_questions;
        const fig1_json = chartsComponent?.fig1_json || '';
        const fig2_json = chartsComponent?.fig2_json || '';
        const dataset = analysisComponent?.dataset;
        const code = analysisComponent?.code || chartsComponent?.code;

        // Extract errors from components
        const hasBusinessError = businessComponent?.status === 'error';
        const hasAnalysisError = analysisComponent?.status === 'error';
        const hasChartsError = chartsComponent?.status === 'error';

        // Extract error details from metadata
        const analysisErrors =
            hasAnalysisError && analysisComponent?.metadata?.exception?.exception_history;
        const analysisAttempts = analysisComponent?.metadata?.attempts;
        const chartsErrors =
            hasChartsError && chartsComponent?.metadata?.exception?.exception_history;
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
            displayDate,
            enhancedUserMessage,
            bottomLine,
            additionalInsights,
            followUpQuestions,
            fig1_json,
            fig2_json,
            dataset,
            code,
            tabStates,
            analysisErrors,
            chartsErrors,
            businessErrors,
            analysisAttempts,
        };
    }, [message, date]);

    return (
        <MessageContainer>
            <MessageHeader avatar={DataRobotAvatar} name="DataRobot" date={displayDate} />

            {isLoading ? (
                <Loading />
            ) : (
                <div className="self-stretch text-sm font-normal leading-tight">
                    {enhancedUserMessage && <div className="mb-3">{enhancedUserMessage}</div>}

                    {message?.error && (
                        <div className="max-h-[300px] overflow-x-auto overflow-y-auto max-w-full">
                            <span className="text-destructive text-sm">{message?.error}</span>
                        </div>
                    )}

                    <ResponseTabs
                        value={activeTab}
                        onValueChange={setActiveTab}
                        tabStates={tabStates}
                    />

                    {activeTab === RESPONSE_TABS.SUMMARY && (
                        <>
                            {chartsErrors && !analysisErrors && (
                                <ErrorPanel errors={chartsErrors} componentType="Charts" />
                            )}
                            <SummaryTabContent
                                bottomLine={bottomLine}
                                fig1={fig1_json}
                                fig2={fig2_json}
                            />
                        </>
                    )}

                    {activeTab === RESPONSE_TABS.INSIGHTS && (
                        <>
                            {businessErrors && (
                                <ErrorPanel
                                    errors={businessErrors}
                                    componentType="Business Insights"
                                />
                            )}
                            <InsightsTabContent
                                additionalInsights={additionalInsights}
                                followUpQuestions={followUpQuestions}
                                chatId={chatId}
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
                            <CodeTabContent dataset={dataset} code={code} />
                        </>
                    )}
                </div>
            )}
        </MessageContainer>
    );
};
