import { useGetSupportedDataSourceTypes } from '@/api/datasets/hooks';
import { useListAvailableDataStores } from '@/api/datasources/hooks';
import { Label } from '@/components/ui/label';
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group';
import { DATA_SOURCES, NEW_DATA_STORE } from '@/constants/dataSources';
import { useTranslation } from '@/i18n';
import { Loader2 } from 'lucide-react';
interface DataSourceSelectorProps {
  value: string;
  onChange: (value: string) => void;
}

export const DataSourceSelector: React.FC<DataSourceSelectorProps> = ({ value, onChange }) => {
  const { t } = useTranslation();
  const dataSources = useGetSupportedDataSourceTypes();
  const availableExternalDataStores = useListAvailableDataStores();

  return (
    <RadioGroup value={value} onValueChange={onChange}>
      <div className="flex items-center space-x-2">
        <RadioGroupItem value={DATA_SOURCES.FILE} id="r1" />
        <Label htmlFor="r1">{t('Local file or Data Registry')}</Label>
      </div>
      <div className="flex items-center space-x-2">
        <RadioGroupItem value={DATA_SOURCES.REMOTE_CATALOG} id="r2" />
        <Label htmlFor="r2">{t('Remote Data Registry')}</Label>
        {dataSources?.isLoading && <Loader2 className="h-4 w-4 animate-spin" />}
      </div>
      {/* Not yet putting a conditional here, though probably in a future release. */}
      <div className="flex items-center space-x-2">
        <RadioGroupItem value={DATA_SOURCES.DATABASE} id="r3" />
        <Label htmlFor="r3">{t('Database')}</Label>
      </div>
      <div className="flex items-center space-x-2">
        <RadioGroupItem value={NEW_DATA_STORE} id="r4" />
        <Label htmlFor="r4">{t('Remote Data Connections')}</Label>
        {availableExternalDataStores?.isLoading && <Loader2 className="h-4 w-4 animate-spin" />}
      </div>
    </RadioGroup>
  );
};
