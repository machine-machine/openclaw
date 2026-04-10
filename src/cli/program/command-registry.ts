import type { Command } from "commander";
import {
  getCoreCliCommandDescriptors,
  getCoreCliCommandNames,
  getCoreCliCommandsWithSubcommands,
  type CommandRegistration,
  registerCoreCliByName,
  registerCoreCliCommands,
} from "./command-registry-core.js";
import type { ProgramContext } from "./context.js";
import {
  getCoreCliCommandDescriptors,
  getCoreCliCommandNames as getCoreDescriptorNames,
  getCoreCliCommandsWithSubcommands,
} from "./core-command-descriptors.js";
import {
  registerCommandGroupByName,
  registerCommandGroups,
  type CommandGroupEntry,
} from "./register-command-groups.js";
import { registerSubCliCommands } from "./register.subclis.js";

export {
  getCoreCliCommandDescriptors,
  getCoreCliCommandNames,
  getCoreCliCommandsWithSubcommands,
  registerCoreCliByName,
  registerCoreCliCommands,
};
export type { CommandRegistration };

export function registerProgramCommands(
  program: Command,
  ctx: ProgramContext,
  argv: string[] = process.argv,
) {
  registerCoreCliCommands(program, ctx, argv);
  registerSubCliCommands(program, argv);
}
