import { EXTERNAL_DATA_STORE_PREFIX } from '@/constants/dataSources';

export function friendlySourceName(dataSourceName: string): string {
  if (dataSourceName && dataSourceName.startsWith(EXTERNAL_DATA_STORE_PREFIX)) {
    return dataSourceName.slice(EXTERNAL_DATA_STORE_PREFIX.length);
  }
  return dataSourceName;
}
