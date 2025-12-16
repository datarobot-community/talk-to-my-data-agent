import { cn } from '~/lib/utils';

export type SidebarMenuOptionType = {
  name: string;
  icon?: React.ReactNode;
  endIcon?: React.ReactNode;
  disabled?: boolean;
  testId?: string;
  id?: string;
  key?: string;
};

type Props = {
  options?: SidebarMenuOptionType[];
  activeKey?: string;
  onClick: (data: SidebarMenuOptionType) => void;
};

export const SidebarMenu = ({ options = [], activeKey, onClick }: Props) => {
  return (
    <div className="flex flex-col gap-2">
      {options.map(option => (
        <SidebarMenuOption
          key={option.key}
          id={option.key}
          name={option.name}
          icon={option.icon}
          endIcon={option.endIcon}
          active={activeKey === option.key}
          disabled={option.disabled}
          onClick={onClick}
          testId={option.testId}
        />
      ))}
    </div>
  );
};

const SidebarMenuOption = ({
  id,
  name,
  icon,
  active,
  disabled,
  onClick,
  testId,
  endIcon,
}: SidebarMenuOptionType & { active: boolean; onClick: (data: SidebarMenuOptionType) => void }) => {
  return (
    <div
      data-testid={testId}
      role="link"
      tabIndex={0}
      onKeyDown={e => {
        if (e.key === 'Enter' || (e.key === ' ' && !disabled)) {
          onClick({ name, id, icon, endIcon, disabled, testId });
        }
      }}
      className={cn(
        'flex gap-2 pr-3 pl-2 py-2 rounded border-l-2 border-transparent overflow-hidden transition-colors cursor-pointer hover:bg-card',
        {
          'rounded-l-none border-l-2 border-l-accent bg-card': active,
          'opacity-50 cursor-not-allowed': disabled,
        }
      )}
      onClick={
        !disabled ? () => onClick({ name, id, icon, endIcon, disabled, testId }) : () => null
      }
    >
      <div className="flex items-center min-w-0 leading-[20px]" title={name}>
        {icon && <div className="flex items-center flex-shrink-0">{icon}</div>}
        <div className="truncate min-w-0 flex-1">{name}</div>
      </div>
      {endIcon && <div className="flex items-center flex-shrink-0">{endIcon}</div>}
    </div>
  );
};
