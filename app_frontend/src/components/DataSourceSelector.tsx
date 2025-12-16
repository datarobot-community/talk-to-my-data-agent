import { useGetSupportedDataSourceTypes } from '@/api/datasets/hooks';
import { useListAvailableDataStores } from '@/api/datasources/hooks';
import { Label } from '@/components/ui/label';
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group';
import { DATA_SOURCES, NEW_DATA_STORE } from '@/constants/dataSources';
import { useTranslation } from '@/i18n';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import { Loader2, MessageCircleQuestion } from 'lucide-react';
interface DataSourceSelectorProps {
  value: string;
  onChange: (value: string) => void;
}

export const DataSourceSelector: React.FC<DataSourceSelectorProps> = ({ value, onChange }) => {
  const { t } = useTranslation();
  const dataSources = useGetSupportedDataSourceTypes();
  const availableExternalDataStores = useListAvailableDataStores();

  return (
    <TooltipProvider>
      <RadioGroup value={value} onValueChange={onChange}>
        <div className="flex space-x-2">
          <RadioGroupItem value={DATA_SOURCES.FILE} id="r1" />
          <Label htmlFor="r1" className="font-bold">
            {t('Local file or Data Registry')}
          </Label>
          <Tooltip>
            <TooltipTrigger asChild>
              <MessageCircleQuestion className="h-3 w-3 text-muted-foreground" />
            </TooltipTrigger>
            <TooltipContent>
              <p className="text-muted-foreground font-normal">
                {t(
                  'Select to upload a local file or your DataRobot Data Registry file up to 200 MB.'
                )}
              </p>
            </TooltipContent>
          </Tooltip>
        </div>
        <div className="flex space-x-2">
          <RadioGroupItem value={DATA_SOURCES.REMOTE_CATALOG} id="r2" />
          <Label htmlFor="r2" className="font-bold">
            {t('Remote Data Registry')}
          </Label>
          <Tooltip>
            <TooltipTrigger asChild>
              <MessageCircleQuestion className="h-3 w-3 text-muted-foreground" />
            </TooltipTrigger>
            <TooltipContent>
              <p className="text-muted-foreground font-normal">
                {t(
                  'Connect and add data from your remote data registry for files greater than 200 MB, up to a maximum of 10 GB. Processing large files may involve lengthy runtimes and increased costs.'
                )}
              </p>
            </TooltipContent>
          </Tooltip>
          {dataSources?.isLoading && <Loader2 className="h-4 w-4 animate-spin" />}
        </div>
        {/* Not yet putting a conditional here, though probably in a future release. */}
        <div className="flex space-x-2">
          <RadioGroupItem value={DATA_SOURCES.DATABASE} id="r3" />
          <Label htmlFor="r3" className="font-bold">
            {t('Database')}
          </Label>
          <Tooltip>
            <TooltipTrigger asChild>
              <MessageCircleQuestion className="h-3 w-3 text-muted-foreground" />
            </TooltipTrigger>
            <TooltipContent>
              <p className="text-muted-foreground font-normal">
                {t(
                  'Select tables from the application-configured database. This option supports Snowflake, SAP DataSphere, and BigQuery.'
                )}
              </p>
            </TooltipContent>
          </Tooltip>
        </div>
        <div className="flex space-x-2">
          <RadioGroupItem value={NEW_DATA_STORE} id="r4" />
          <Label htmlFor="r4" className="font-bold">
            {t('Remote Data Connections')}
          </Label>
          <Tooltip>
            <TooltipTrigger asChild>
              <MessageCircleQuestion className="h-3 w-3 text-muted-foreground" />
            </TooltipTrigger>
            <TooltipContent>
              <p className="text-muted-foreground font-normal">
                {t(
                  'Select tables from DataRobot supported data store connections, such as Redshift or PostgreSQL.'
                )}
              </p>
            </TooltipContent>
          </Tooltip>
          {availableExternalDataStores?.isLoading && <Loader2 className="h-4 w-4 animate-spin" />}
        </div>
      </RadioGroup>
    </TooltipProvider>
  );
};
