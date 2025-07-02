import React from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faPaperPlane } from '@fortawesome/free-solid-svg-icons/faPaperPlane';
import { useGeneratedDictionaries } from '@/api/dictionaries/hooks';
import { usePostMessage } from '@/api/chat-messages/hooks';
import { useAppState } from '@/state/hooks';

interface SuggestedPromptProps {
    message: string;
    chatId?: string;
}

export const SuggestedPrompt: React.FC<SuggestedPromptProps> = ({ message, chatId }) => {
    const { enableChartGeneration, enableBusinessInsights, dataSource } = useAppState();
    const { data: dictionaries } = useGeneratedDictionaries();
    const isDisabled = !dictionaries?.[0];
    const { mutate } = usePostMessage();
    return (
        <div className="h-16 p-3 bg-[#22272b] rounded border justify-start items-center gap-2 inline-flex">
            <div className="grow shrink basis-0 text-primary text-sm font-normal leading-tight">
                {message}
            </div>
            <div className="w-9 h-9 p-2 justify-center items-center flex">
                <div className="w-5 h-5 flex-col justify-center items-center gap-2.5 inline-flex">
                    <div className="text-center text-sm leading-tight cursor-pointer">
                        {!isDisabled && (
                            <FontAwesomeIcon
                                icon={faPaperPlane}
                                onClick={() => {
                                    mutate({
                                        message,
                                        chatId,
                                        enableChartGeneration,
                                        enableBusinessInsights,
                                        dataSource,
                                    });
                                }}
                            />
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
};
