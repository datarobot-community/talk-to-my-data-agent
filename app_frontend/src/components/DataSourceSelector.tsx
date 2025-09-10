import { Label } from '@/components/ui/label';
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group';
import { DATA_SOURCES } from '@/constants/dataSources';
import { useTranslation } from '@/i18n';
interface DataSourceSelectorProps {
  value: string;
  onChange: (value: string) => void;
}

export const DataSourceSelector: React.FC<DataSourceSelectorProps> = ({ value, onChange }) => {
  const { t } = useTranslation();
  return (
    <RadioGroup value={value} onValueChange={onChange}>
      <div className="flex items-center space-x-2">
        <RadioGroupItem value={DATA_SOURCES.FILE} id="r1" />
        <Label htmlFor="r1">{t('Local file or Data Registry')}</Label>
      </div>
      <div className="flex items-center space-x-2">
        <RadioGroupItem value={DATA_SOURCES.REMOTE_CATALOG} id="r1" />
        <Label htmlFor="r1">{t('Remote Data Registry')}</Label>
      </div>
      <div className="flex items-center space-x-2">
        <RadioGroupItem value={DATA_SOURCES.DATABASE} id="r2" />
        <Label htmlFor="r2">{t('Database')}</Label>
      </div>
    </RadioGroup>
  );
};
