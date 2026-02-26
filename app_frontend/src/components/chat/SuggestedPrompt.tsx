import React from 'react';
import { Send, Hourglass } from 'lucide-react';
import { useGeneratedDictionaries } from '@/api/dictionaries/hooks';
import { usePostMessage } from '@/api/chat-messages/hooks';
import { useAppState } from '@/state/hooks';
import { useTranslation } from '@/i18n';
import { Button } from '@/components/ui/button';

interface SuggestedPromptProps {
  message: string;
  chatId?: string;
  hasInProgressMessages: boolean;
}

export const SuggestedPrompt: React.FC<SuggestedPromptProps> = ({
  message,
  chatId,
  hasInProgressMessages,
}) => {
  const { t } = useTranslation();
  const { enableChartGeneration, enableBusinessInsights, dataSource } = useAppState();
  const { data: dictionaries } = useGeneratedDictionaries();
  const { mutate: postMessage } = usePostMessage();
  const isActionShown = !!dictionaries?.[0];
  const actionTooltip = hasInProgressMessages
    ? t('Wait for agent to finish responding')
    : t('Send');

  return (
    <div className="inline-flex h-16 items-center justify-start gap-2 rounded border p-3">
      <div className="shrink grow basis-0 body">{message}</div>
      <div className="flex size-9 items-center justify-center p-2">
        <div className="inline-flex size-5 flex-col items-center justify-center gap-2.5">
          <div className="cursor-pointer text-center text-sm leading-tight">
            {isActionShown && (
              <Button
                variant="ghost"
                testId="send-suggested-prompt-button"
                disabled={hasInProgressMessages}
                title={actionTooltip}
                onClick={() => {
                  postMessage({
                    message,
                    chatId,
                    enableChartGeneration,
                    enableBusinessInsights,
                    dataSource,
                  });
                }}
              >
                {hasInProgressMessages ? <Hourglass /> : <Send />}
              </Button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
