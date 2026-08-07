import { describe, test, expect } from 'vitest';
import { browseData, constants } from '@datarobot/connectivity';

import { isDatabaseTableSubmit } from '@/lib/browse-data';

const { DATA_STORE_TYPES } = constants;
const { DATA_REGISTRY_STORE } = browseData;

/**
 * These guard DM-21892: the submit routing compared `storeType` against
 * `DATA_STORE_TYPES.NATIVE_DATABASE`, a key the package does not define. It
 * evaluated to `undefined`, so every store with no `type` — the Data Registry
 * above all — matched and was sent to the data-store PUT, which 404s.
 *
 * `DATA_STORE_TYPES` is typed as `Record<string, DataStoreType>`, so tsc cannot
 * catch a bad key. Assert the contract we depend on instead.
 */
describe('connectivity constants contract', () => {
  test('exposes the database store types the submit routing keys off', () => {
    expect(DATA_STORE_TYPES.JDBC).toBe('jdbc');
    expect(DATA_STORE_TYPES.NATIVE).toBe('dr-database-v1');
  });

  test('has no NATIVE_DATABASE key', () => {
    expect(DATA_STORE_TYPES.NATIVE_DATABASE).toBeUndefined();
  });

  test('the Data Registry pseudo-store has an id but no type', () => {
    expect(DATA_REGISTRY_STORE.id).toBeTruthy();
    expect(DATA_REGISTRY_STORE).not.toHaveProperty('type');
  });
});

describe('isDatabaseTableSubmit', () => {
  test.each([
    ['jdbc', DATA_STORE_TYPES.JDBC],
    ['native database', DATA_STORE_TYPES.NATIVE],
  ])('routes a %s table pick to the data-store path', (_label, storeType) => {
    expect(isDatabaseTableSubmit({ storeId: 'store-1', storeType, isSqlQuery: false })).toBe(true);
  });

  test('routes a Data Registry selection to the catalog path', () => {
    expect(
      isDatabaseTableSubmit({
        storeId: DATA_REGISTRY_STORE.id,
        // The registry pseudo-store carries no type — the case that regressed.
        storeType: undefined,
        isSqlQuery: false,
      })
    ).toBe(false);
  });

  test('routes a native connector to the catalog path', () => {
    expect(
      isDatabaseTableSubmit({
        storeId: 'store-1',
        storeType: DATA_STORE_TYPES.CONNECTOR,
        isSqlQuery: false,
      })
    ).toBe(false);
  });

  test('routes a SQL query off a database connection to the catalog path', () => {
    expect(
      isDatabaseTableSubmit({
        storeId: 'store-1',
        storeType: DATA_STORE_TYPES.JDBC,
        isSqlQuery: true,
      })
    ).toBe(false);
  });

  test('is false without a store id', () => {
    expect(
      isDatabaseTableSubmit({
        storeId: undefined,
        storeType: DATA_STORE_TYPES.JDBC,
        isSqlQuery: false,
      })
    ).toBe(false);
  });
});
