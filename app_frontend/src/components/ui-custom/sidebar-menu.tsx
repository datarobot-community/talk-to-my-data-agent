import { cn } from '~/lib/utils';

export type SidebarMenuOptionType = {
  name: string;
  subtitle?: string;
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
  subtitle,
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
          onClick({ name, subtitle, id, icon, endIcon, disabled, testId });
        }
      }}
      className={cn(
        'flex cursor-pointer gap-2 overflow-hidden rounded border-l-2 border-transparent py-2 pr-3 pl-2 transition-colors hover:bg-card',
        {
          'rounded-l-none border-l-2 border-l-accent bg-card': active,
          'cursor-not-allowed opacity-50': disabled,
        }
      )}
      onClick={
        !disabled
          ? () => onClick({ name, subtitle, id, icon, endIcon, disabled, testId })
          : () => null
      }
    >
      <div className="flex min-w-0 items-center leading-[20px]" title={name}>
        {icon && <div className="flex flex-shrink-0 items-center">{icon}</div>}
        <div className="min-w-0 flex-1">
          <div className="truncate">{name}</div>
          {subtitle && <div className="truncate text-xs text-muted-foreground">{subtitle}</div>}
        </div>
      </div>
      {endIcon && <div className="flex flex-shrink-0 items-center">{endIcon}</div>}
    </div>
  );
};
