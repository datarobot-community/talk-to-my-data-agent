import React from "react";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faQuoteLeft } from "@fortawesome/free-solid-svg-icons/faQuoteLeft";
import { faTable } from "@fortawesome/free-solid-svg-icons/faTable";
import { DATA_TABS } from "@/state/constants";
import { ValueOf } from "@/state/types";

interface DataViewTabsProps {
  defaultValue?: ValueOf<typeof DATA_TABS>;
  onChange?: (value: ValueOf<typeof DATA_TABS>) => void;
}

export const DataViewTabs: React.FC<DataViewTabsProps> = ({
  defaultValue = DATA_TABS.DESCRIPTION,
  onChange,
}) => {
  return (
    <Tabs
      defaultValue={defaultValue}
      onValueChange={onChange}
      className="w-fit py-4"
    >
      <TabsList className="grid w-full grid-cols-2">
        <TabsTrigger
          value={DATA_TABS.DESCRIPTION}
          data-testid="description-tab"
        >
          <FontAwesomeIcon className="mr-2" icon={faQuoteLeft} />
          Description
        </TabsTrigger>
        <TabsTrigger value={DATA_TABS.RAW} data-testid="raw-tab">
          <FontAwesomeIcon className="mr-2" icon={faTable} />
          Raw rows
        </TabsTrigger>
      </TabsList>
    </Tabs>
  );
};
