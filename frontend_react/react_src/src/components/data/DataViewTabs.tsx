import React from "react";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faQuoteLeft } from "@fortawesome/free-solid-svg-icons/faQuoteLeft";
import { faTable } from "@fortawesome/free-solid-svg-icons/faTable";
import { DATA_TABS } from "@/state/constants";
import { ValueOf } from "@/state/types";
import { useTranslation } from "react-i18next";

interface DataViewTabsProps {
  defaultValue?: ValueOf<typeof DATA_TABS>;
  onChange?: (value: ValueOf<typeof DATA_TABS>) => void;
}

export const DataViewTabs: React.FC<DataViewTabsProps> = ({
  defaultValue = DATA_TABS.DESCRIPTION,
  onChange,
}) => {
  const { t } = useTranslation();
  return (
    <Tabs 
      defaultValue={defaultValue} 
      onValueChange={onChange}
      className="w-fit py-4"
    >
      <TabsList className="grid w-full grid-cols-2">
        <TabsTrigger value={DATA_TABS.DESCRIPTION}>
          <FontAwesomeIcon className="mr-2" icon={faQuoteLeft} />
          {t("description")}
        </TabsTrigger>
        <TabsTrigger value={DATA_TABS.RAW}>
          <FontAwesomeIcon className="mr-2" icon={faTable} />
          {t("raw_rows")}
        </TabsTrigger>
      </TabsList>
    </Tabs>
  );
};
