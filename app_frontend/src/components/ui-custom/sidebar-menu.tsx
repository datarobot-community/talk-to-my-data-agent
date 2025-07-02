import { cn } from "~/lib/utils";

export type SidebarMenuOptionType = {
  key: string;
  name: string;
  icon?: React.ReactNode;
  active?: boolean;
  disabled?: boolean;
  onClick: () => void;
  testId?: string;
};

type Props = {
  options: SidebarMenuOptionType[];
  activeKey?: string;
};

export const SidebarMenu = ({ options }: Props) => {
  return (
    <div className="flex flex-col gap-2">
      {options.map((option) => (
        <SidebarMenuOption
          key={option.key}
          name={option.name}
          icon={option.icon}
          active={option.active}
          disabled={option.disabled}
          onClick={option.onClick}
          testId={option.testId}
        />
      ))}
    </div>
  );
};

const SidebarMenuOption = ({
  name,
  icon,
  active,
  disabled,
  onClick,
  testId,
}: SidebarMenuOptionType) => {
  return (
    <div
      data-testid={testId}
      role="link"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === "Enter" || (e.key === " " && !disabled)) {
          onClick();
        }
      }}
      className={cn(
        "flex gap-2 px-3 py-2 rounded border-l-2 border-transparent overflow-hidden transition-colors cursor-pointer hover:bg-card",
        {
          "rounded-l-none border-l-2 border-primary bg-card": active,
          "opacity-50 cursor-not-allowed": disabled,
        }
      )}
      onClick={!disabled ? onClick : () => null}
    >
      <div className="break-words flex my-1" title={name}>
        {icon && <div className="flex items-center">{icon}</div>}
        <div className="flex ml-2">{name}</div>
      </div>
    </div>
  );
};
