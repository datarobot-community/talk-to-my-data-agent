import React from 'react';
import { HeaderSection } from './HeaderSection';
import { PlotPanel } from './PlotPanel';
import { parsePlotData } from './utils';
import { useTranslation } from '@/i18n';

interface SummaryTabContentProps {
  bottomLine?: string;
  fig1: string;
  fig2: string;
}

export const SummaryTabContent: React.FC<SummaryTabContentProps> = ({ bottomLine, fig1, fig2 }) => {
  const { t } = useTranslation();
  const plot1 = parsePlotData(fig1);
  const plot2 = parsePlotData(fig2);

  return (
    <div>
      {bottomLine && <HeaderSection title={t('Bottom line')}>{bottomLine}</HeaderSection>}
      <div className="flex flex-col gap-2.5">
        {plot1 && <PlotPanel plotData={plot1} />}
        {plot2 && <PlotPanel plotData={plot2} />}
      </div>
    </div>
  );
};
