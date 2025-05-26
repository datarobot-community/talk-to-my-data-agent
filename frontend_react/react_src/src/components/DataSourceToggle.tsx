import React, { useMemo } from "react";
import { useAppState } from "@/state/hooks";
import { DATA_SOURCES } from "@/constants/dataSources";
import { DatasetMetadata } from "@/api-state/cleansed-datasets/api-requests";
import { ToggleGroup, ToggleGroupItem } from "./ui/toggle-group";
import {
  useFetchAllChats,
  useUpdateChatDataSource,
} from "@/api-state/chat-messages/hooks";
import { useParams } from "react-router-dom";
import { useTranslation } from "react-i18next";

interface IDataSourceToggleProps {
  multipleMetadata?: {
    name: string;
    metadata: DatasetMetadata;
  }[];
}

/**
 * Toggle component for switching between database and catalog data sources
 */
export const DataSourceToggle: React.FC<IDataSourceToggleProps> = ({
  multipleMetadata,
}) => {
  const { chatId } = useParams<{ chatId?: string }>();
  const { dataSource, setDataSource } = useAppState();
  const { data: chats } = useFetchAllChats();
  const { mutate: updateChatDataSource } = useUpdateChatDataSource();
  const { t } = useTranslation();

  const handleValueChange = (value: string) => {
    if (value) {
      if (chatId) {
        updateChatDataSource({ chatId, dataSource: value });
      }
      setDataSource(value);
    }
  };

  // Get current value - either from the chat or from global state
  const currentValue = chatId
    ? chats?.find((c) => c.id === chatId)?.data_source || dataSource
    : dataSource;

  const dataByDataSource = useMemo(() => {
    return multipleMetadata?.reduce((acc, { name, metadata }) => {
      const { data_source } = metadata;
      if (
        currentValue === DATA_SOURCES.FILE &&
        (data_source === DATA_SOURCES.FILE ||
          data_source === DATA_SOURCES.CATALOG)
      ) {
        acc.push(name);
      }

      if (
        currentValue === DATA_SOURCES.DATABASE &&
        data_source === DATA_SOURCES.DATABASE
      ) {
        acc.push(name);
      }
      return acc;
    }, [] as string[]);
  }, [multipleMetadata, currentValue]);

  const getTooltip = () => {
    return (
      <div className="absolute min-w-[270px] max-w-[600px] left-1/2 -translate-x-1/2 top-full mb-2 hidden bg-secondary text-secondary-foreground group-hover:block text-xs rounded px-2 py-1 shadow z-10">
        {t("selected_data_sources", "Selected data sources:")}
        {dataByDataSource && dataByDataSource?.length > 0 ? (
          <ul className="list-disc pl-5">
            {dataByDataSource.map((item) => (
              <li key={item} className="text-sm">
                {item}
              </li>
            ))}
          </ul>
        ) : (
          <div className="text-sm">{t("no_data_sources_selected", "No data sources selected.")}</div>
        )}
      </div>
    );
  };

  return (
    <ToggleGroup
      type="single"
      value={currentValue}
      onValueChange={handleValueChange}
      className="bg-muted rounded-md p-1 shadow-sm group relative"
    >
      {getTooltip()}
      <ToggleGroupItem value={DATA_SOURCES.DATABASE} className="text-sm">
        <div className="m-2">{t("database")}</div>
      </ToggleGroupItem>
      <ToggleGroupItem value={DATA_SOURCES.FILE} className="text-sm">
        <div className="m-2">{t("registry_file")}</div>
      </ToggleGroupItem>
    </ToggleGroup>
  );
};
