import { Toaster as Sonner, type ToasterProps } from 'sonner';
import { XCircle, CheckCircle, AlertCircle, Info } from 'lucide-react';
import { useTheme } from '@/theme/theme-provider';

const Toaster = ({ ...props }: ToasterProps) => {
  const { theme } = useTheme();

  return (
    <Sonner
      theme={theme}
      richColors
      expand={true}
      duration={10000}
      visibleToasts={4}
      icons={{
        error: <XCircle className="size-4" />,
        success: <CheckCircle className="size-4" />,
        warning: <AlertCircle className="size-4" />,
        info: <Info className="size-4" />,
      }}
      {...props}
    />
  );
};

export { Toaster };
