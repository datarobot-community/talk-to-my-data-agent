import { Toaster as Sonner, ToasterProps } from 'sonner';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import {
  faCircleXmark,
  faCircleCheck,
  faCircleExclamation,
  faCircleInfo,
} from '@fortawesome/free-solid-svg-icons';

const Toaster = ({ ...props }: ToasterProps) => {
  return (
    <Sonner
      theme="dark"
      richColors
      expand={true}
      duration={10000}
      visibleToasts={4}
      icons={{
        error: <FontAwesomeIcon icon={faCircleXmark} />,
        success: <FontAwesomeIcon icon={faCircleCheck} />,
        warning: <FontAwesomeIcon icon={faCircleExclamation} />,
        info: <FontAwesomeIcon icon={faCircleInfo} />,
      }}
      {...props}
    />
  );
};

export { Toaster };
