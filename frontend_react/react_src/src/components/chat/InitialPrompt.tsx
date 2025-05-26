import { useState, useMemo } from "react";
import { PromptInput } from "@/components/ui-custom/prompt-input";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faPaperPlane } from "@fortawesome/free-solid-svg-icons/faPaperPlane";
import chatMidnight from "@/assets/chat-midnight.svg";
import {
  usePostMessage,
  useFetchAllChats,
} from "@/api-state/chat-messages/hooks";
import { useAppState } from "@/state/hooks";
import { DATA_SOURCES } from "@/constants/dataSources";
import { useTranslation } from "react-i18next";

export const InitialPrompt = ({
  chatId,
  allowedDataSources,
}: {
  allowedDataSources?: string[];
  chatId?: string;
}) => {
  const {
    enableChartGeneration,
    enableBusinessInsights,
    dataSource: globalDataSource,
  } = useAppState();
  const { data: chats } = useFetchAllChats();
  const { mutate } = usePostMessage();
  const [message, setMessage] = useState("");
  const isDisabled = !allowedDataSources?.[0];
  const { t } = useTranslation();

  // Find the active chat to get its data source setting
  const activeChat = chatId
    ? chats?.find((chat) => chat.id === chatId)
    : undefined;
  const chatDataSource = useMemo(() => {
    const dataSource = activeChat?.data_source || globalDataSource;
    // User can only select from the allowed data sources
    return allowedDataSources?.includes(dataSource)
      ? dataSource
      : allowedDataSources?.[0] || DATA_SOURCES.FILE;
  }, [activeChat?.data_source, globalDataSource, allowedDataSources]);

  const sendMessage = () => {
    if (message.trim()) {
      mutate({
        message,
        chatId,
        enableChartGeneration,
        enableBusinessInsights,
        dataSource: chatDataSource,
      });
      setMessage("");
    }
  };

  return (
    <div className="flex-1 flex flex-col p-4">
      <div className="flex flex-col flex-1 items-center justify-center">
        <div className="w-[400px] flex flex-col flex-1 items-center justify-center">
          <img src={chatMidnight} alt="" />
          <h4 className="mb-2 mt-4">
            <strong className=" text-center font-semibold">
              {t("prompt_title")}
            </strong>
          </h4>
          <p className="text-center mb-10">
            {t("prompt_description")}
          </p>
          <PromptInput
            icon={FontAwesomeIcon}
            iconProps={{
              icon: isDisabled ? null : faPaperPlane,
              behavior: "append",
              onClick: sendMessage,
            }}
            disabled={isDisabled}
            aria-disabled={isDisabled}
            onChange={(e) => setMessage(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                sendMessage();
              }
            }}
            value={message}
            placeholder={
              isDisabled
                ? t("prompt_placeholder_disabled")
                : t("prompt_placeholder_enabled")
            }
          />
        </div>
      </div>
    </div>
  );
};
