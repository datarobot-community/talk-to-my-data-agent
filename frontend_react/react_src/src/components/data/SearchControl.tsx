import React from "react";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faMagnifyingGlass } from "@fortawesome/free-solid-svg-icons/faMagnifyingGlass";
import { useTranslation } from "react-i18next";

interface SearchControlProps {
  onSearch?: (searchText: string) => void;
}

export const SearchControl: React.FC<SearchControlProps> = () => {
  const { t } = useTranslation();
  return (
    <div>
      <FontAwesomeIcon className="mx-2" icon={faMagnifyingGlass} />
      {t("search")}
    </div>
  );
};