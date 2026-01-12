import React, { useState, useRef, useEffect } from 'react';
import { Separator } from '@radix-ui/react-separator';
import { useGeneratedDictionaries } from '@/api/dictionaries/hooks';
import { useNavigate } from 'react-router-dom';
import { generateDataRoute } from '@/pages/routes';

import { DatasetCardDescriptionPanel, DataViewTabs, ClearDatasetsButton } from '@/components/data';
import { ValueOf } from '@/state/types';
import { useTranslation } from '@/i18n';
import { DATA_TABS } from '@/state/constants';
import { Loading } from '@/components/ui-custom/loading';
import { useLocation, useParams } from 'react-router';
import { useDebounce, cn } from '@/lib/utils';

export const Data: React.FC = () => {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const location = useLocation();
  const { dataId } = useParams();
  const { data, status } = useGeneratedDictionaries();
  const [viewMode, setViewMode] = useState<ValueOf<typeof DATA_TABS>>(DATA_TABS.DESCRIPTION);
  const ref = useRef<{ [key: string]: HTMLDivElement }>({});
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (location.state?.cancelScroll && dataId) {
      navigate(generateDataRoute(dataId), { state: null, replace: true });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if ((!dataId && data?.length) || (data?.length && !data.find(({ name }) => dataId === name))) {
      navigate(generateDataRoute(data[0].name), { state: null, replace: true });
    }
    if (dataId && data?.length === 0) {
      navigate(generateDataRoute(), { state: null });
    }
  }, [dataId, data, navigate]);

  useEffect(() => {
    if (ref.current && dataId && data?.length && !location.state?.cancelScroll) {
      ref.current[dataId]?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [dataId, data, location.state?.cancelScroll]);

  const handleScroll = useDebounce(() => {
    const containerTop = containerRef.current?.getBoundingClientRect().top || 0;
    const data = Object.keys(ref.current);
    for (const key of data) {
      const element = ref.current[key];
      const elementTop = element.getBoundingClientRect().top;
      if (elementTop - containerTop >= 0) {
        navigate(generateDataRoute(key), { state: { cancelScroll: true } });
        break;
      }
    }
  }, 100);

  return (
    <div className="p-6 pr-0 flex flex-col h-full">
      <h2 className="heading-04">{t('Data')}</h2>
      <div className="flex justify-between gap-2">
        <div className="flex gap-2 items-center">
          <div className="text-sm">{t('View')}</div>
          <DataViewTabs
            defaultValue={viewMode}
            onChange={value => setViewMode(value as ValueOf<typeof DATA_TABS>)}
          />
        </div>
        <div className="flex items-center">{!!data?.length && <ClearDatasetsButton />}</div>
      </div>
      <Separator className="my-4 border-t" />
      {status === 'pending' ? (
        <div className="flex items-center justify-center h-[calc(100vh-200px)]">
          <Loading />
        </div>
      ) : (
        <div
          ref={containerRef}
          onScroll={handleScroll}
          className={cn('flex flex-1 flex-col gap-4 pr-6', {
            'overflow-y-auto': data && data.length > 1,
            'overflow-hidden': data && data.length === 1,
          })}
        >
          {data?.map(dictionary => (
            <DatasetCardDescriptionPanel
              ref={element => {
                if (element) {
                  ref.current[dictionary.name] = element;
                }
              }}
              fullHeight={data?.length === 1}
              key={dictionary.name}
              isProcessing={dictionary.in_progress || false}
              dictionary={dictionary}
              viewMode={viewMode}
            />
          ))}
        </div>
      )}
    </div>
  );
};
