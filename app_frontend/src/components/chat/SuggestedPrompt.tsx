import React from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faPaperPlane } from '@fortawesome/free-solid-svg-icons/faPaperPlane';
import { faHourglassHalf } from '@fortawesome/free-solid-svg-icons/faHourglassHalf';
import { useGeneratedDictionaries } from '@/api/dictionaries/hooks';
import { usePostMessage } from '@/api/chat-messages/hooks';
import { useAppState } from '@/state/hooks';
import { useTranslation } from '@/i18n';
import { Button } from '@/components/ui/button';

interface SuggestedPromptProps {
  message: string;
  chatId?: string;
  isProcessing?: boolean;
}

export const SuggestedPrompt: React.FC<SuggestedPromptProps> = ({
  message,
  chatId,
  isProcessing,
}) => {
  const { t } = useTranslation();
  const { enableChartGeneration, enableBusinessInsights, dataSource } = useAppState();
  const { data: dictionaries } = useGeneratedDictionaries();
  const isActionShown = !!dictionaries?.[0];
  const actionTooltip = isProcessing ? t('Wait for agent to finish responding') : t('Send');
  const { mutate } = usePostMessage();
  return (
    <div className="h-16 p-3 bg-[#22272b] rounded border justify-start items-center gap-2 inline-flex">
      <div className="grow shrink basis-0 text-primary text-sm font-normal leading-tight">
        {message}
      </div>
      <div className="w-9 h-9 p-2 justify-center items-center flex">
        <div className="w-5 h-5 flex-col justify-center items-center gap-2.5 inline-flex">
          <div className="text-center text-sm leading-tight cursor-pointer">
            {isActionShown && (
              <Button
                variant="ghost"
                testId="send-suggested-prompt-button"
                disabled={isProcessing}
                title={actionTooltip}
                onClick={() => {
                  mutate({
                    message,
                    chatId,
                    enableChartGeneration,
                    enableBusinessInsights,
                    dataSource,
                  });
                }}
              >
                <FontAwesomeIcon icon={isProcessing ? faHourglassHalf : faPaperPlane} />
              </Button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
