#include<stdio.h>
enum ret_codes {A,forward, backward,left,right,LB,RB,none};

int off_state(void)
{
    int command;
    printf("enter command (OFF) : \n");
    scanf("%d", &command);
    return command;
    
}
int stand_state(int command)
{
    int command;
    printf("enter command (STAND) : \n");
    scanf("%d", &command);
    return command;
}
int stomp_state(int command)
{
    int command;
    printf("enter command (STOMP) : \n");
    scanf("%d", &command);
    return command;
}
int turn_left_state(void)
{
    int command;
    printf("enter command (TURN_LEFT) : \n");
    scanf("%d", &command);
    return command;
}
int turn_right_state(void)
{
   int command;
    printf("enter command (TURN_RIGHT) : \n");
    scanf("%d", &command);
    return command;
}
int trot_forward_state(void)
{
    int command;
    printf("enter command (TROT_FORWARD) : \n");
    scanf("%d", &command);
    return command;
}
int trot_backward_state(void)
{
    int command;
    printf("enter command (TROT_BACKWARD) : \n");
    scanf("%d", &command);
    return command;
}
int sidestep_left_state(void)
{
    int command;
    printf("enter command (SIDESTEP_LEFT) : \n");
    scanf("%d", &command);
    return command;

}
int sidestep_right_state(void)
{
    int command;
    printf("enter command (SIDESTEP_RIGHT) : \n");
    scanf("%d", &command);
    return command;
}


/* array and enum below must be in sync! */
int (* state[])(void) = { off_state, stand_state, stomp_state, turn_left_state, turn_right_state, trot_forward_state, trot_backward_state, sidestep_left_state, sidestep_right_state};
enum state_codes {off, stand, stomp, turn_left, turn_right, trot_forward, trot_backward, sidestep_left, sidestep_right};

struct transition {
    enum state_codes src_state;
    enum ret_codes   ret_code;
    enum state_codes dst_state;
};
/* transitions from end state aren't needed */
struct transition state_transitions[] = 
{
    {off, A, stand},
    
    {stand, forward, trot_forward},
    {stand, backward, trot_backward},
    {stand, left, sidestep_left},
    {stand, right, sidestep_right},
    {stand, LB, turn_left},
    {stand, RB, turn_right},

    {trot_forward, none, stomp},
    {trot_backward, none, stomp},
    {turn_left, none, stomp},
    {turn_right, none, stomp},
    {sidestep_left, none, stomp},
    {sidestep_right, none, stomp},



};

#define EXIT_STATE end
#define ENTRY_STATE entry

int lookup_transitions(int cur_state, int rc)
{
    for(int i =0; i < 8; i++)
    {
        if(state_transitions[i].src_state == cur_state && state_transitions[i].ret_code == rc)
        {
            return state_transitions[i].dst_state;
        }
    }
    printf("Specified transition does not exist");
    return 1000;
}
int main(int argc, char *argv[]) {
    enum state_codes cur_state = ENTRY_STATE;
    enum ret_codes rc;
    int (* state_fun)(void);

    for (;;) {
        state_fun = state[cur_state];
        rc = state_fun();
        if (EXIT_STATE == cur_state)
            break;
        cur_state = lookup_transitions(cur_state, rc);
    }

}