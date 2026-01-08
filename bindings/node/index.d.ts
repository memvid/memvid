/** Options for the put operation. */
export interface PutInput {
  /** The text content to store. */
  text: string;
  /** Optional title for the document. */
  title?: string;
  /** Optional URI identifier. */
  uri?: string;
  /** Optional list of tags. */
  tags?: string[];
  /** Optional list of labels. */
  labels?: string[];
  /** Optional Unix timestamp. */
  timestamp?: number;
}

/**
 * A portable AI memory file (.mv2).
 *
 * Use `Memvid.create()` to create a new memory file or `Memvid.open()` to open an existing one.
 */
export class Memvid {
  /** Create a new memory file at the given path. */
  static create(path: string): Memvid;

  /** Open an existing memory file. */
  static open(path: string): Memvid;

  /** Open a memory file in read-only mode. */
  static openReadOnly(path: string): Memvid;

  /**
   * Store content and return the frame ID.
   *
   * @param input - The content to store with optional metadata.
   * @returns The frame ID that can be used with remove().
   */
  put(input: PutInput): number;

  /**
   * Remove a frame by its ID.
   *
   * This is a soft delete - the frame is marked as deleted and removed from indexes.
   *
   * @param frameId - The frame ID returned by put().
   * @returns The WAL sequence number of the delete operation.
   */
  remove(frameId: number): number;

  /** Commit pending changes to disk. */
  commit(): void;

  /** Seal the memory file (commit and close). */
  seal(): void;

  /** Get the number of frames in the memory. */
  frameCount(): number;

  /** Check if the memory is read-only. */
  isReadOnly(): boolean;
}
