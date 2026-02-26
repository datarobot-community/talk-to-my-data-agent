import React, { Suspense, lazy } from 'react';
import { CollapsiblePanel } from './CollapsiblePanel';
import { PlotlyData } from './utils';
import './PlotPanel.css';
import { useTranslation } from '@/i18n';

const Plot = lazy(() => {
  return import('react-plotly.js');
});

const PlotLoading = () => {
  const { t } = useTranslation();
  return (
    <div className="flex h-[200px] items-center justify-center p-4">
      <div>{t('Loading visualization...')}</div>
    </div>
  );
};

interface PlotPanelProps {
  plotData: {
    data: PlotlyData[];
    layout: {
      title?: {
        text?: string;
      };
      [key: string]: unknown;
    };
  };
  className?: string;
  width?: string | number;
  height?: string | number;
}

export const PlotPanel: React.FC<PlotPanelProps> = ({
  plotData,
  className = '',
  width = '100%',
  height = '500px',
}) => {
  if (!plotData || !plotData.data || !plotData.layout) {
    return null;
  }

  return (
    <CollapsiblePanel header={plotData.layout?.title?.text}>
      <Suspense fallback={<PlotLoading />}>
        <Plot
          data={plotData.data}
          layout={plotData.layout}
          className={className}
          style={{ position: 'relative', width, height }}
          config={{ responsive: true }}
        />
      </Suspense>
    </CollapsiblePanel>
  );
};
