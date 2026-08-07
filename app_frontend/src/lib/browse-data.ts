import { constants } from '@datarobot/connectivity';

const { DATA_STORE_TYPES } = constants;

/**
 * True when a Browse Data submit is a database table pick, which registers
 * directly against the connection instead of going through a catalog item.
 *
 * Only `jdbc` and `dr-database-v1` stores qualify. Everything else — the Data
 * Registry (a pseudo-store carrying no `type` at all), native connectors, and
 * uploads — must take the catalog path, as must any SQL-query selection, which
 * has no catalog/schema/table to register.
 *
 * The `storeType` guard is load-bearing: this compared against a
 * `DATA_STORE_TYPES.NATIVE_DATABASE` that the package never defined, so it read
 * as `undefined` and swallowed every typeless store — sending Data Registry
 * items to the data-store PUT, which 404s. DM-21892.
 */
export function isDatabaseTableSubmit({
  storeId,
  storeType,
  isSqlQuery,
}: {
  storeId?: string;
  storeType?: string;
  isSqlQuery: boolean;
}): boolean {
  if (!storeId || isSqlQuery || !storeType) return false;
  return storeType === DATA_STORE_TYPES.JDBC || storeType === DATA_STORE_TYPES.NATIVE;
}
