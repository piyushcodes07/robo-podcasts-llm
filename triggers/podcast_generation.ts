
import { task } from "@trigger.dev/sdk/v3";
import { python } from "@trigger.dev/python";
import { z } from "zod";

export const podcastGeneration = task({
  id: "podcast-generation",
  run: async (payload: any, { ctx }) => {
    const result = await python.runScript("./podcast_llm/generate.py", {
      args: [
        payload.topic,
        `--main_user_id=${payload.main_user_id}`,
        `--mode=${payload.mode}`,
        `--sources=${payload.sources}`,
        `--qa_rounds=${payload.qa_rounds}`,
        `--audio_output=${payload.audio_output}`,
        `--text_output=${payload.text_output}`,
        `--user_id=${payload.user_id}`,
      ],
    });

    if (result.exitCode !== 0) {
      throw new Error(`Python script failed with exit code ${result.exitCode}`);
    }

    return result.stdout;
  },
});
