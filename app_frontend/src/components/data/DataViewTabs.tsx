import React from 'react';
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Quote, Table } from 'lucide-react';
import { DATA_TABS } from '@/state/constants';
import { useTranslation } from '@/i18n';

import { ValueOf } from '@/state/types';

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
      data-testid="data-view-tabs"
      defaultValue={defaultValue}
      onValueChange={onChange}
      className="w-fit py-4"
    >
      <TabsList>
        <TabsTrigger value={DATA_TABS.DESCRIPTION} data-testid="description-tab">
          <Quote className="mr-2 size-4" />
          {t('Description')}
        </TabsTrigger>
        <TabsTrigger value={DATA_TABS.RAW} data-testid="raw-tab">
          <Table className="mr-2 size-4" />
          {t('Raw rows')}
        </TabsTrigger>
      </TabsList>
    </Tabs>
  );
};
