"use client";

import React from "react";
import clsx from "clsx";

export interface IssueDetailTab {
  id: string;                  // e.g. "overview", "ai-fix", "metadata", "raw"
  label: string;               // display label
  icon?: React.ReactNode;      // optional icon (lucide, etc.)
  badgeCount?: number;         // e.g. number of sources / metadata fields
  disabled?: boolean;
}

export interface IssueDetailTabsProps {
  tabs: IssueDetailTab[];
  activeId: string;
  onChange: (id: string) => void;
  className?: string;
}

/* -------------------------------------------------------------------------- */
/*                                  Component                                  */
/* -------------------------------------------------------------------------- */

const IssueDetailTabs: React.FC<IssueDetailTabsProps> = ({
  tabs,
  activeId,
  onChange,
  className,
}) => {
  if (!tabs || tabs.length === 0) return null;

  return (
    <div className={clsx("border-b border-neutral-200", className)}>
      <div className="flex gap-2 overflow-x-auto scrollbar-none">
        {tabs.map((tab) => {
          const isActive = tab.id === activeId;
          const isDisabled = tab.disabled;

          return (
            <button
              key={tab.id}
              type="button"
              disabled={isDisabled}
              onClick={() => !isDisabled && onChange(tab.id)}
              className={clsx(
                "relative inline-flex items-center gap-2 px-3 py-2 text-sm font-medium border-b-2",
                "transition-colors whitespace-nowrap",
                isActive
                  ? "border-neutral-900 text-neutral-900"
                  : "border-transparent text-neutral-500 hover:text-neutral-800 hover:border-neutral-200",
                isDisabled && "opacity-50 cursor-not-allowed"
              )}
            >
              {/* Icon */}
              {tab.icon && (
                <span className="flex items-center text-neutral-500">
                  {tab.icon}
                </span>
              )}

              {/* Label */}
              <span>{tab.label}</span>

              {/* Badge */}
              {typeof tab.badgeCount === "number" && tab.badgeCount > 0 && (
                <span
                  className={clsx(
                    "inline-flex items-center justify-center text-[11px] font-semibold",
                    "px-1.5 py-0.5 rounded-full bg-neutral-100 text-neutral-700"
                  )}
                >
                  {tab.badgeCount}
                </span>
              )}
            </button>
          );
        })}
      </div>
    </div>
  );
};

export { IssueDetailTabs };
export default IssueDetailTabs;
