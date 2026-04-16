import logging

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    TurnHandlingOptions,
    cli,
)
from livekit.plugins import elevenlabs, openai, silero, speechmatics

logger = logging.getLogger("local-agent")

load_dotenv()


class LocalAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a helpful voice AI assistant. "
                "Keep your responses concise and conversational. "
                "Do not use emojis, asterisks, markdown, or special characters."
            ),
        )

    async def on_enter(self) -> None:
        self.session.generate_reply(
            instructions="Greet the user and ask how you can help."
        )


server = AgentServer()


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    session = AgentSession(
        stt=speechmatics.STT(),
        llm=openai.LLM(model="gpt-4o"),
        tts=elevenlabs.TTS(voice_id="21m00Tcm4TlvDq8ikWAM"),
        vad=ctx.proc.userdata["vad"],
        turn_handling=TurnHandlingOptions(
            turn_detection="stt",
            endpointing={
                "min_delay": 1.0,
                "max_delay": 5.0,
            },
            interruption={
                "mode": "vad",
                "resume_false_interruption": True,
            },
        ),
    )

    await session.start(
        agent=LocalAgent(),
        room=ctx.room,
    )


if __name__ == "__main__":
    cli.run_app(server)
